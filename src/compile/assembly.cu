
__global__ void copy_u8(char *in, char *out) {
  int d;
  asm("ld.u8 %0, [%1];" : "=r"(d) : "l"(in));
  *out = d;
}

__global__ void add_f32(int len, uint *out) {
  uint c = 0;
#pragma unroll(1)
  for (uint i = 0; i < len; i++) {
    asm("add.u32 %0,%1,%2;" : "+r"(c) : "r"(i), "r"(c));
  }
  *out = c;
}

__global__ void LD1Impl(const float *__restrict__ input, float *output,
                        int len) {
  float x;
  float y = 0;
#pragma unroll(1)
  for (size_t i = 0; i < len; i++) {
    const auto *ptr = input + i;
    __asm__ __volatile__("ld.global.f32 %0,[%1];" : "=f"(x) : "l"(ptr));
    y += x;
  }
  output[0] = y;
}

__global__ void LD4Impl(const float *__restrict__ input, float *output,
                        int len) {
  float4 v4;
  float4 y4;
#pragma unroll(1)
  for (size_t i = 0; i < len; i += 4) {
    const auto *ptr = input + i;
    __asm__ __volatile__("ld.global.v4.f32 {%0,%1,%2,%3},[%4];"
                         : "=f"(v4.x), "=f"(v4.y), "=f"(v4.z), "=f"(v4.w)
                         : "l"(ptr));
    y4.x += v4.x;
    y4.y += v4.y;
    y4.z += v4.z;
    y4.w += v4.w;
  }
  output[0] = v4.x;
  output[1] = v4.y;
  output[2] = v4.z;
  output[3] = v4.w;
}