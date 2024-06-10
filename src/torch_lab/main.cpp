#include <torch/torch.h>
#include <iostream>

int main() {
    // Create a tensor and print its size
    torch::Tensor tensor = torch::eye(3);
    std::cout << "Tensor size: " << tensor.sizes() << std::endl;
    return 0;
}