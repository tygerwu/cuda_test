#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
// supported type: b,B,i,u
template <typename To, typename From, typename = typename std::enable_if<std::is_convertible<From, To>::value>::type>
static std::vector<To> ConvertType(const std::vector<From> &fromVec) {
    std::vector<To> toVec;
    std::transform(fromVec.begin(), fromVec.end(), std::back_inserter(toVec),
                   [](const From &from) { return (To)(from); });
    return toVec;
}

class NpyArray {
    friend class NpzDict;

   public:
    template <typename T, typename U, typename = typename std::enable_if<std::is_convertible<U, size_t>::value>::type>
    NpyArray(const T *data, const std::vector<U> &storageShape, const std::string &name = {});
    NpyArray(const NpyArray &other) { *this = other; }
    NpyArray(NpyArray &&other) { *this = std::move(other); }
    NpyArray() = default;
    ~NpyArray() {
        if (data_) {
            free(data_);
        }
    }

   public:
    static NpyArray Load(const std::string &filePath);

   public:
    template <typename T>
    const T *Data() const {
        return static_cast<T *>(data_);
    }
    template <typename T>
    T *Data() {
        return static_cast<T *>(data_);
    }
    template <typename T>
    std::shared_ptr<T> CopiedData() const;
    void Save(const std::string &filePath);
    size_t Size() const;
    size_t Bytes() const { return Size() * typeSize_; }
    size_t TypeSize() const { return typeSize_; }
    const std::vector<size_t> &StorageShape() const { return storageShape_; }
    const std::vector<size_t> &Shape() const { return shape_; }
    void SetName(const std::string &name) { name_ = name; }
    const std::string GetName() const { return name_; }
    NpyArray &operator=(const NpyArray &other);
    NpyArray &operator=(NpyArray &&other);

   private:
    static NpyArray LoadImpl(std::ifstream &f);
    static NpyArray LoadImpl(const char *buf);
    static std::vector<size_t> ParseShape(const std::string &shapeStr);
    static std::string DefaultName();
    static void ParseHeader(const std::string &header, size_t &typeSize, std::vector<size_t> &shape,
                            std::string &numpyType, bool &needSwapEndian);
    template <typename T>
    std::string ToNumpyType() const;
    std::string ToShapeStr() const;
    std::vector<char> Serialize() const;
    std::vector<char> SerializeHeader() const;

   private:
    void *data_ = nullptr;
    std::vector<size_t> shape_ = {};
    size_t typeSize_;
    std::string numpyType_;
    std::string name_;
    std::vector<size_t> storageShape_ = {};
};

class NpzDict {
   public:
    NpzDict() = default;
    NpzDict(const std::vector<NpyArray> &npzDict) : npzDict_(npzDict) {}
   

   public:
    static NpzDict Load(const std::string &filePath);
    void Save(const std::string &filePath);
    template <typename T,
              typename = typename std::enable_if<std::is_same<typename std::decay<T>::type, NpyArray>::value>::type>
    void Add(T &&npyArray, bool overWrite = false) {
        bool found = false;
        if (overWrite) {
            for (auto &array : npzDict_) {
                if (array.name_ == npyArray.name_) {
                    array = std::forward<T>(npyArray);
                    found = true;
                    break;
                }
            }
        }
        if (!found || !overWrite) {
            npzDict_.push_back(std::forward<T>(npyArray));
        }
    }
    const std::vector<NpyArray> &Values() const { return npzDict_; }

   public:
    // Mock the behavior of a dict
    const NpyArray *operator[](const std::string &key) const {
        auto iter =
            std::find_if(npzDict_.begin(), npzDict_.end(), [&](const NpyArray &array) { return array.name_ == key; });
        if (iter != npzDict_.end()) {
            return &(*iter);
        }
        return nullptr;
    }
    const std::vector<const NpyArray *> operator[](const std::vector<std::string> &keys) const {
        std::vector<const NpyArray *> res;
        for (const auto &key : keys) {
            auto array = (*this)[key];
            if (!array) {
                return {};
            } else {
                res.push_back((*this)[key]);
            }
        }
        return res;
    }
    NpyArray *operator[](const std::string &key) {
        auto iter =
            std::find_if(npzDict_.begin(), npzDict_.end(), [&](const NpyArray &array) { return array.name_ == key; });
        if (iter != npzDict_.end()) {
            return &(*iter);
        }
        return nullptr;
    }

   private:
    static void Decompress(const void *before, void *after, size_t beforeBytes, size_t afterBytes);
    std::vector<char> Compress(const NpyArray &array, size_t &unCompressedSize, uint32_t &crc);
    std::vector<char> CreateLocalHeader(const std::string &fileName, const NpyArray &array, size_t unCompressedSize,
                                        size_t compressedSize, uint32_t crc);
    std::vector<char> CreateCentralEntry(const std::vector<char> &localHeader, size_t localHeaderOffset,
                                         const std::string &fName);
    std::vector<char> CreateFooter(size_t entryNum, size_t centralDirSize, size_t centralDirOffset);

   private:
    // Use vector to reserve the order in which the NpyArrays(Tensors) were added
    std::vector<NpyArray> npzDict_;
};
template <typename T>
std::shared_ptr<T> NpyArray::CopiedData() const {
    size_t bytes = Bytes();
    void *data = malloc(bytes);
    ::memcpy(data, data_, bytes);
    std::shared_ptr<T> res(data, [](T *p) { free(p); });
    return res;
}
template <typename T, typename U, typename>
NpyArray::NpyArray(const T *data, const std::vector<U> &storageShape, const std::string &name) {
    storageShape_ = std::move(ConvertType<size_t>(storageShape));
    shape_ = storageShape_;
    typeSize_ = sizeof(T);
    size_t bytes = Bytes();
    data_ = malloc(bytes);
    numpyType_ = ToNumpyType<T>();
    if (name.empty()) {
        name_ = DefaultName();
    }
    name_ = name;
    ::memcpy(data_, data, bytes);
}

template <typename T>
std::string NpyArray::ToNumpyType() const {
    // if (std::is_same<T, float16_t>::value) {
    //     return "f2";
    // }
    if (std::is_same<T, float>::value) {
        return "f4";
    } else if (std::is_same<T, double>::value) {
        return "f8";
    } else if (std::is_same<T, char>::value) {
        return "i1";
    } else if (std::is_same<T, short>::value) {
        return "i2";
    } else if (std::is_same<T, int>::value) {
        return "i4";
    } else if (std::is_same<T, short>::value) {
        return "i8";
    } else if (std::is_same<T, unsigned char>::value) {
        return "u1";
    } else if (std::is_same<T, unsigned short>::value) {
        return "u2";
    } else if (std::is_same<T, unsigned int>::value) {
        return "u4";
    } else if (std::is_same<T, unsigned long>::value) {
        return "u8";
    } else {
        RUNTIME_ASSERT(false, "Unsupported type");
    }
}
