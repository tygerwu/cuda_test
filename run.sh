set -x

FILTER=cute.tensor2

cd ./vs_build_Debug && cmake --build . -t all -j 12

./cuda_lab_test --gtest_filter=${FILTER}


cd ..