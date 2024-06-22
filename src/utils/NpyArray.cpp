#include "NpyArray.hpp"
#include <zconf.h>
#include <zlib.h>
#include <numeric>
#include <regex>
#include <string>

template <typename T>
static void Append(std::vector<char> &vec, T u) {
    char *p = (char *)&u;
    for (size_t i = 0; i < sizeof(T); i++) {
        vec.push_back(p[i]);
    }
}
static void Append(std::vector<char> &vec, const std::string &str) {
    std::copy(str.begin(), str.end(), std::back_inserter(vec));
}
static void Append(std::vector<char> &vec, std::string &&str) {
    std::copy(str.begin(), str.end(), std::back_inserter(vec));
}
static void Append(std::vector<char> &vec, char c) { vec.push_back(c); }

static void Append(std::vector<char> &vec, const std::vector<char> &other) {
    std::copy(other.begin(), other.end(), std::back_inserter(vec));
}

template <typename T>
static void Append(std::string &str, T u) {
    char *p = (char *)&u;
    for (size_t i = 0; i < sizeof(T); i++) {
        str.push_back(p[i]);
    }
}
static bool IsBigEndian() {
    int a = 1;
    char *p = (char *)(&a);
    return *p == 0;
}

template <typename T>
static T SwapEndian(T u) {
    union {
        T num;
        unsigned char mem[sizeof(T)];
    } source, dest;
    source.num = u;
    size_t size = sizeof(T);
    for (size_t i = 0; i < size; i++) {
        dest.mem[i] = source.mem[size - 1 - i];
    }
    return dest.num;
}

static void SwapEndian(const char *src, char *dst, size_t totalBytes, size_t typeSize) {
    RUNTIME_ASSERT(totalBytes % typeSize == 0, "Invalid totalBytes or typeSize");
    for (size_t i = 0; i < totalBytes; i += typeSize) {
        for (size_t j = 0; j < typeSize; j++) {
            dst[i + j] = src[i + typeSize - 1 - j];
        }
    }
}

// Swap in place
static void SwapEndian(char *data, size_t totalBytes, size_t typeSize) {
    RUNTIME_ASSERT(totalBytes % typeSize == 0, "Invalid totalBytes or typeSize");
    std::vector<char> tmp(typeSize);
    for (size_t i = 0; i < totalBytes; i += typeSize) {
        for (size_t j = 0; j < typeSize; j++) {
            tmp[j] = data[i + typeSize - 1 - j];
        }
        ::memcpy(tmp.data(), data + i, typeSize);
    }
}

#pragma mark NpyArray

std::string NpyArray::DefaultName() {
    static int nameId = 0;
    return std::string("NpyArray") + std::to_string(nameId++);
}


NpyArray &NpyArray::operator=(const NpyArray &other) {
    if (data_) {
        free(data_);
        data_ = nullptr;
    }
    storageShape_ = other.storageShape_;
    shape_ = other.shape_;
    typeSize_ = other.typeSize_;
    numpyType_ = other.numpyType_;
    size_t bytes = Bytes();
    data_ = malloc(bytes);
    name_ = other.name_;
    ::memcpy(data_, other.data_, bytes);
    return *this;
}
NpyArray &NpyArray::operator=(NpyArray &&other) {
    storageShape_ = std::move(other.storageShape_);
    shape_ = std::move(other.shape_);
    typeSize_ = other.typeSize_;
    numpyType_ = std::move(other.numpyType_);
    if (data_) {
        free(data_);
        data_ = nullptr;
    }
    name_ = std::move(other.name_);
    data_ = other.data_;
    other.data_ = nullptr;
    return *this;
}

size_t NpyArray::Size() const {
    return std::accumulate(storageShape_.begin(), storageShape_.end(), static_cast<size_t>(1),
                           std::multiplies<size_t>());
}

//()
//(20,)
//(20, 30)
//(20, 30, 40)
std::string NpyArray::ToShapeStr() const {
    // scalar
    if (shape_.size() == 0) {
        return "()";
    }
    std::string tmp("(");
    size_t size = storageShape_.size();
    if (storageShape_.size() == 1) {
        return (tmp + std::to_string(storageShape_[0]) + ",)");
    } else {
        for (size_t i = 0; i < size - 1; i++) {
            tmp += std::to_string(storageShape_[i]);
            tmp += ", ";
        }
        return (tmp + std::to_string(storageShape_.back()) + ")");
    }
}

void NpyArray::Save(const std::string &filePath) {
    std::ofstream f;
    f.open(filePath, std::ofstream::out | std::ofstream::trunc);
    RUNTIME_ASSERT(f.is_open(), "File error");
    auto header = SerializeHeader();
    f.write(header.data(), header.size());
    RUNTIME_ASSERT(f.good(), "File error");
    f.write((char *)data_, Bytes());
    RUNTIME_ASSERT(f.good(), "File error");
    f.close();
    return;
}

void NpyArray::ParseHeader(const std::string &header, size_t &typeSize, std::vector<size_t> &shape,
                           std::string &numpyType, bool &needSwapEndian) {
    //"{'descr': '<i2', 'fortran_order': False, 'shape': (60,), } "
    std::smatch result;
    RUNTIME_ASSERT(std::regex_search(header, result,
                                     std::regex(".*(<|>|\\|)(.*)',.*:[[:space:]](False|"
                                                "True).*[[:space:]](\\(.*\\)),")),
                   "Regex error in ParseHeader");

    std::string endian = result[1];
    if ((endian == "<" && IsBigEndian()) || (endian == ">" && !IsBigEndian())) {
        needSwapEndian = true;
    }

    // type size
    numpyType = result[2];
    RUNTIME_ASSERT(numpyType.size() == 2, "Invalid type in ParseHeader");
    typeSize = std::stoi(numpyType.substr(1, 1));

    // no fortran order
    std::string fortran = result[3];
    RUNTIME_ASSERT(fortran == "False", "Invalid fortran order");
    // shape
    std::string shapeStr = result[4];
    shape = ParseShape(shapeStr);
}

NpyArray NpyArray::LoadImpl(std::ifstream &f) {
    const size_t metaLen = 10;
    std::vector<char> buf(metaLen);
    f.read(buf.data(), metaLen);
    RUNTIME_ASSERT(f.good(), "Fail to load metaData");

    // Parse 9th and 10th to get header len
    uint16_t headerLen = *((uint16_t *)(buf.data() + 8));
    if (IsBigEndian()) {
        headerLen = SwapEndian<uint16_t>(headerLen);
    }
    buf.resize(headerLen);
    f.read(buf.data(), headerLen);
    RUNTIME_ASSERT(f.good(), "Fail to load header");
    std::string header(buf.data(), headerLen);

    size_t typeSize;
    std::vector<size_t> shape;
    std::string numpyType;
    bool needSwapEndian = false;
    ParseHeader(header, typeSize, shape, numpyType, needSwapEndian);

    NpyArray npyArray;
    size_t bytes =
        std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) * typeSize;

    npyArray.data_ = malloc(bytes);
    f.read((char *)npyArray.data_, bytes);
    RUNTIME_ASSERT(f.good(), "Fail to load body");
    if (needSwapEndian) {
        SwapEndian((char *)npyArray.data_, bytes, typeSize);
    }
    npyArray.shape_ = shape;
    npyArray.storageShape_ = std::move(shape);
    npyArray.typeSize_ = typeSize;
    npyArray.numpyType_ = std::move(numpyType);
    npyArray.name_ = DefaultName();
    return npyArray;
}

NpyArray NpyArray::LoadImpl(const char *buf) {
    const size_t headerLenOffset = 8;
    const size_t metaLen = 10;
    // ToDo : endian
    int16_t headerLen = *((int16_t *)(buf + headerLenOffset));
    std::string header(buf + headerLenOffset, headerLen);

    size_t typeSize;
    std::vector<size_t> shape;
    std::string numpyType;
    bool needSwapEndian = false;
    ParseHeader(header, typeSize, shape, numpyType, needSwapEndian);

    NpyArray npyArray;
    size_t bytes =
        std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) * typeSize;

    npyArray.data_ = malloc(bytes);
    if (needSwapEndian) {
        SwapEndian((char *)buf + metaLen + headerLen, (char *)npyArray.data_, bytes, typeSize);
    } else {
        memcpy(npyArray.data_, buf + metaLen + headerLen, bytes);
    }
    npyArray.shape_ = shape;
    npyArray.storageShape_ = std::move(shape);
    npyArray.typeSize_ = typeSize;
    npyArray.numpyType_ = std::move(numpyType);
    return npyArray;
}

NpyArray NpyArray::Load(const std::string &filePath) {
    std::ifstream f(filePath, std::ios::binary);
    RUNTIME_ASSERT(f.good(), "Fail to load file");
    auto array = LoadImpl(f);
    f.close();
    return array;
}

//(1,)
//(2, 2)
//()
std::vector<size_t> NpyArray::ParseShape(const std::string &shapeStr) {
    //()
    if (shapeStr.size() == 2) {
        return {1};
    }
    size_t last = 1;
    std::vector<size_t> shape;
    char c = ',';
    while (true) {
        size_t found = shapeStr.find(c, last);
        if (found != std::string::npos) {
            std::string numStr = shapeStr.substr(last, found - last);
            shape.push_back(std::stoi(numStr));
            last = found + 2;
        } else {
            if (c == ',') {
                c = ')';
            } else {
                break;
            }
        }
    }
    return shape;
}

std::vector<char> NpyArray::Serialize() const {
    auto res = SerializeHeader();
    std::copy((char *)data_, (char *)data_ + Bytes(), std::back_inserter(res));
    return res;
}
std::vector<char> NpyArray::SerializeHeader() const {
    std::vector<char> headerDict;
    //{'descr': '<i2', 'fortran_order': False, 'shape': (60,), }
    Append(headerDict, std::string("{'descr': '"));
    if (typeSize_ == 1) {
        Append(headerDict, '|');
    } else {
        if (IsBigEndian()) {
            Append(headerDict, '>');
        } else {
            Append(headerDict, '<');
        }
    }
    Append(headerDict, numpyType_);
    Append(headerDict, std::string("', 'fortran_order': False, 'shape': "));
    Append(headerDict, ToShapeStr());
    Append(headerDict, std::string(", }"));

    size_t alignedLen = UP_ROUND(headerDict.size() + 10, 64);
    uint16_t headerLen = static_cast<uint16_t>(alignedLen - 10);
    if (IsBigEndian()) {
        headerLen = SwapEndian<uint16_t>(headerLen);
    }

    std::string padding(headerLen - (headerDict.size() + 1), ' ');
    padding.append("\n");

    std::vector<char> header;
    Append(header, (uint8_t)0x93);
    Append(header, std::string("NUMPY"));
    Append(header, (uint8_t)0x01);
    Append(header, (uint8_t)0x00);
    Append(header, (uint16_t)headerLen);
    Append(header, headerDict);
    Append(header, padding);

    return header;
}

#pragma mark NpzDict

// Decompress the data in a single step(use Z_FINISH)
void NpzDict::Decompress(const void *in, void *out, size_t inBytes, size_t outBytes) {
    int status;
    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    RUNTIME_ASSERT(inflateInit2(&strm, -MAX_WBITS) == Z_OK, "InflateInit error");
    strm.avail_in = static_cast<unsigned int>(inBytes);
    strm.avail_out = static_cast<unsigned int>(outBytes);
    strm.next_in = &(((unsigned char *)in)[0]);
    strm.next_out = &(((unsigned char *)out)[0]);

    status = inflate(&strm, Z_FINISH);
    RUNTIME_ASSERT(status == Z_OK || status == Z_STREAM_END, "Inflate error");

    inflateEnd(&strm);
}

NpzDict NpzDict::Load(const std::string &filePath) {
    std::ifstream f(filePath, std::ios::binary);
    RUNTIME_ASSERT(f.good(), "File error");
    NpzDict res;
    std::vector<char> buf(30);
    const size_t generalFlagOffset = 6;
    const size_t compressMethodOffset = 8;
    const size_t compressedSizeOffset = 18;
    const size_t unCompressedSizeOffset = 22;
    const size_t fileNameLenOffset = 26;
    const size_t extraFieldLenOffset = 28;
    const size_t dataDescBytes = 12;
    char *ptr = buf.data();
    std::vector<char> fileNameBuf;
    std::vector<char> compressed;
    std::vector<char> unCompressed;
    while (true) {
        // local header
        f.read(ptr, 30);
        if (!f.good()) {
            break;
        }
        // local header identifier
        if (ptr[0] == 0x50 && ptr[1] == 0x4b && ptr[2] == 0x03 && ptr[3] == 0x04) {
            // general flag
            uint16_t generalFlag = *(uint16_t *)(ptr + generalFlagOffset);
            // compress method
            uint16_t compressMethod = *(uint16_t *)(ptr + compressMethodOffset);
            // size after being compressed
            uint32_t inSize = *(uint32_t *)(ptr + compressedSizeOffset);
            // uncomressed size
            uint32_t outSize = *(uint32_t *)(ptr + unCompressedSizeOffset);
            // filepath len
            uint16_t fileNameLen = *(uint16_t *)(ptr + fileNameLenOffset);
            // extra len
            uint16_t extraFieldLen = *(uint16_t *)(ptr + extraFieldLenOffset);

            fileNameBuf.resize(fileNameLen);
            // xxxx.npy
            f.read(fileNameBuf.data(), fileNameLen);
            std::string key(fileNameBuf.data(), fileNameLen - 4);
            // seek to data start
            f.seekg(extraFieldLen, std::ios_base::cur);
            if (compressMethod == 0) {
                auto array = NpyArray::LoadImpl(f);

                array.name_ = std::move(key);
                res.Add(std::move(array));
            } else {
                compressed.clear();
                compressed.resize(inSize);
                f.read(compressed.data(), inSize);
                if (!f.good()) {
                    RUNTIME_ASSERT(false, "Fail to decompress");
                }
                unCompressed.resize(outSize);
                Decompress(compressed.data(), unCompressed.data(), inSize, outSize);
                auto array = NpyArray::LoadImpl(unCompressed.data());
                array.name_ = std::move(key);
                res.Add(std::move(array));
            }
            // Skip data desc
            if (generalFlag == 1) {
                f.seekg(dataDescBytes, std::ios_base::cur);
            }
        } else {
            break;
        }
    }
    f.close();
    return res;
}

std::vector<char> NpzDict::Compress(const NpyArray &array, size_t &unCompressedSize, uint32_t &crc) {
    auto data = array.Serialize();
    size_t bytes = data.size();
    unCompressedSize = bytes;
    crc = static_cast<uint32_t>(crc32(0L, (unsigned char *)data.data(), static_cast<unsigned int>(bytes)));
    std::vector<char> out(bytes);

    z_stream strm;
    int status;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    status = deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, MAX_MEM_LEVEL, Z_DEFAULT_STRATEGY);
    RUNTIME_ASSERT(status == Z_OK, "DeflateInit2 error");

    strm.avail_in = static_cast<unsigned int>(bytes);
    strm.next_in = (unsigned char *)(data.data());
    strm.avail_out = static_cast<unsigned int>(bytes);
    strm.next_out = (unsigned char *)out.data();

    status = deflate(&strm, Z_FINISH);
    RUNTIME_ASSERT(status == Z_OK || status == Z_STREAM_END, "Deflate error");
    size_t left = strm.avail_out;
    deflateEnd(&strm);
    out.resize(bytes - left);
    return out;
}

// The data is compressed automatically
void NpzDict::Save(const std::string &filePath) {
    std::ofstream f;
    f.open(filePath, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    RUNTIME_ASSERT(f.good(), "File error");
    std::vector<char> localHeader;
    size_t unCompressedSize;
    std::vector<std::vector<char>> centralEntries;
    for (const auto &array : npzDict_) {
        uint32_t crc;
        auto compressed = Compress(array, unCompressedSize, crc);
        // Build local header
        localHeader = CreateLocalHeader(array.name_, array, unCompressedSize, compressed.size(), crc);
        size_t offset = f.tellp();
        centralEntries.push_back(CreateCentralEntry(localHeader, offset, array.name_ + ".npy"));
        f.write(localHeader.data(), localHeader.size());
        RUNTIME_ASSERT(f.good(), "File error");
        f.write(compressed.data(), compressed.size());
        RUNTIME_ASSERT(f.good(), " File error ");
    }
    size_t centralDirSize = 0;
    size_t centralDirOffset = f.tellp();
    for (const auto &entry : centralEntries) {
        centralDirSize += entry.size();
        f.write(entry.data(), entry.size());
        RUNTIME_ASSERT(f.good(), "File error");
    }
    auto footer = CreateFooter(centralEntries.size(), centralDirSize, centralDirOffset);
    f.write(footer.data(), footer.size());
    RUNTIME_ASSERT(f.good(), "File error");
    f.close();
}

std::vector<char> NpzDict::CreateLocalHeader(const std::string &fileName, const NpyArray &array,
                                             size_t unCompressedSize, size_t compressedSize, uint32_t crc) {
    std::vector<char> localHeader;
    // local header identifier
    Append(localHeader, (uint32_t)0x04034b50);
    // min version to extract
    Append(localHeader, (uint16_t)(20));
    // general purpose bit flag
    Append(localHeader, (uint16_t)(0));
    // Compression Method
    Append(localHeader, (uint16_t)(Z_DEFLATED));
    // last mod file time
    Append(localHeader, (uint16_t)(0));
    // last mod file date
    Append(localHeader, (uint16_t)(0));
    // crc
    Append(localHeader, (uint32_t)(crc));
    // compressed size
    Append(localHeader, (uint32_t)(compressedSize));
    // uncompressed size
    Append(localHeader, (uint32_t)(unCompressedSize));
    // filename len
    Append(localHeader, (uint16_t)(fileName.size() + 4));
    // extra filed length
    Append(localHeader, (uint16_t)(0));
    // filename
    Append(localHeader, (fileName + ".npy"));
    return localHeader;
}

std::vector<char> NpzDict::CreateCentralEntry(const std::vector<char> &localHeader, size_t localHeaderOffset,
                                              const std::string &entryName) {
    std::vector<char> entryHeader;
    //  identifier
    Append(entryHeader, (uint32_t)0x02014b50);
    // version made by
    Append(entryHeader, (uint16_t)20);
    entryHeader.insert(entryHeader.end(), localHeader.begin() + 4, localHeader.begin() + 30);
    // file comment length
    Append(entryHeader, (uint16_t)0);
    // disk number start
    Append(entryHeader, (uint16_t)0);
    //	internal file attributes
    Append(entryHeader, (uint16_t)0);
    // external file attributes
    Append(entryHeader, (uint32_t)0);
    // relative offset of local header
    Append(entryHeader, (uint32_t)localHeaderOffset);
    // file name
    Append(entryHeader, entryName);
    return entryHeader;
}

std::vector<char> NpzDict::CreateFooter(size_t entryNum, size_t centralDirSize, size_t centralDirOffset) {
    std::vector<char> footer;
    // Identifier
    Append(footer, uint32_t(0x06054b50));
    // number of this disk
    Append(footer, uint16_t(0));
    // number of the disk with the start of the central directory
    Append(footer, uint16_t(0));
    // total number of entries in the central directory on this disk
    Append(footer, uint16_t(entryNum));
    // total number of entries in the central directory
    Append(footer, uint16_t(entryNum));
    // size of the central directory
    Append(footer, uint32_t(centralDirSize));
    // offset of start of central directory with respect to the starting disk
    // number
    Append(footer, uint32_t(centralDirOffset));
    // ZIP file comment length
    Append(footer, uint16_t(0));

    return footer;
}
