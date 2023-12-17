

//         - use,
//     -I / usr / local / cuda / include

namespace N1 {
struct S1_t {
  int i;
  double d;
};
__global__ void f3(int *result) { *result = sizeof(T); };

{
#include <cxxabi.h>
  namespace T1 {
  struct S1 {
    int i = 0;
  };
  } // namespace T1
  auto mangled_name = typeid(T1::S1).name();
  int status = 0;
  size_t len = 0;
  char *buf = nullptr;
  auto demangled_name = abi::__cxa_demangle(mangled_name, buf, &len, &status);
}