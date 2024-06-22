#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include <sys/syscall.h>
#include "utils/device.cuh"

DEFINE_int32(profile_loop,1,"Loop Times");
int profile_loop = 1;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  PrintDeviceInfo();
  profile_loop = FLAGS_profile_loop;
  return RUN_ALL_TESTS();
}
