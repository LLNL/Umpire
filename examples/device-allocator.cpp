#include "umpire/ResourceManager.hpp"
#include "umpire/DeviceAllocator.hpp"

__global__
void my_kernel(umpire::DeviceAllocator alloc, double** data_ptr) {
  if (threadIdx.x == 0) {
    double* data = static_cast<double*>(alloc.allocate(10*sizeof(double)));
    *data_ptr = data;

    data[7] = 1024;
  }
}

int main(int argc, char const *argv[]) {

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto device_allocator = umpire::DeviceAllocator(allocator, 1024);

  double** ptr_to_data = static_cast<double**>(allocator.allocate(sizeof(double*)));

  my_kernel<<<1, 16>>>(device_allocator, ptr_to_data);
  cudaDeviceSynchronize();

  std::cout << (*ptr_to_data)[7] << std::endl;
}
