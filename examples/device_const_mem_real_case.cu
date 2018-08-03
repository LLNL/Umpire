#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <string>

#include "umpire/ResourceManager.hpp"
#include <cuda_runtime_api.h>

__global__ void add_constant_kernel(const double *a, double *d, double *c, int size)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size)
  {
    c[i] = a[i] + d[i];
  }
}

__global__ void add_kernel(const double *a, const double *b, double *c, int size)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
    c[i] = a[i] + b[i];
}

void check_error(void)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(err);
  }
}

int main(int, char**)
{
  // auto& rm = umpire::ResourceManager::getInstance();

  const int size = 1024 * 8;

  auto& rm = umpire::ResourceManager::getInstance();

  std::cout << "Available allocators: ";
  for (auto s : rm.getAvailableAllocators()){
    std::cout << s << "  ";
  }
  std::cout << std::endl;

  // double *a = (double*)malloc(sizeof(double) * size);
  // double *b = (double*)malloc(sizeof(double) * size);
  // double *sum = (double*)malloc(sizeof(double) * size);

  auto host_alloc = rm.getAllocator("HOST");
  double *sum = static_cast<double*>(host_alloc.allocate(size*sizeof(double)));
  double *a = static_cast<double*>(host_alloc.allocate(size*sizeof(double)));
  double *b = static_cast<double*>(host_alloc.allocate(size*sizeof(double)));
  std::cout << "Host memory allocation finished\n";

  // double *d_a;
  // double *d_b;
  // double *d_sum;

  auto dev_alloc = rm.getAllocator("DEVICE");
  double *d_sum = static_cast<double*>(dev_alloc.allocate(size*sizeof(double)));
  double *d_a = static_cast<double*>(dev_alloc.allocate(size*sizeof(double)));
  double *d_b = static_cast<double*>(dev_alloc.allocate(size*sizeof(double)));
  std::cout << "Device memory allocation finished\n";


  auto dev_const_alloc = rm.getAllocator("DEVICE_CONST");
  double *d_d = static_cast<double*>(dev_const_alloc.allocate(size*sizeof(double)));
  std::cout << "Device constant memory allocation finished\n";



  for (int i=0; i<size; i++)
  {
  	a[i] = static_cast<double>(i);
  	b[i] = a[i];
    sum[i] = 1;
  }

  // cudaMemcpy(d_a, a, size*sizeof(double), cudaMemcpyHostToDevice);
  rm.copy(d_a, a, size*sizeof(double));
  check_error();

    // cudaMemcpy(d_b, b, size*sizeof(double), cudaMemcpyHostToDevice);
  rm.copy(d_b, b, size*sizeof(double));
  check_error();

  // cudaMemcpy(d_d, b, size*sizeof(double), cudaMemcpyHostToDevice);
  rm.copy(d_d, b, size*sizeof(double));
  check_error();

  std::cout << "Memory copy finished\n";

// ----------------add kernel-------------------------
  add_kernel<<<256, 1024>>>(d_a, d_b, d_sum, size);
  check_error();

  // cudaMemcpy(sum, d_sum, size*sizeof(double), cudaMemcpyDeviceToHost);
  rm.copy(sum, d_sum, size*sizeof(double));
  check_error();
  
  int final_sum = 0;
  for (int i=0; i<size; i++)
  {
    final_sum += sum[i];	 
  }
  std::cout << "device memory result: " << final_sum << std::endl;

// ----------------add const kernel-------------------------
  add_constant_kernel<<<256, 1024>>>(d_a, d_d, d_sum, size);
  check_error();

  // cudaMemcpy(sum, d_sum, size*sizeof(double), cudaMemcpyDeviceToHost);
  rm.copy(sum, d_sum, size*sizeof(double));
  check_error();

  int final_sum_const = 0;
  for (int i=0; i<size; i++)
  {
     final_sum_const += sum[i];
  }
  std::cout << "const memory result:  " << final_sum_const << std::endl;

  if (final_sum_const - final_sum == 0)
    std::cout << "Same result, success" << std::endl;
  else
    std::cout << "Wrong result" << std::endl;

  // cudaFree(d_a);
  // cudaFree(d_b);
  // cudaFree(d_sum);
  // free(a);
  // free(b);
  // free(sum);

  dev_alloc.deallocate(d_a);
  dev_alloc.deallocate(d_b);
  dev_alloc.deallocate(d_sum);
  host_alloc.deallocate(a);
  host_alloc.deallocate(b);
  host_alloc.deallocate(sum);
  dev_const_alloc.deallocate(d_d);

  return 0;
}