//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <assert.h>
#include <iostream>
#include <sstream>

__global__ void tester(size_t* d_data, size_t size)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
   
  if (idx == 0) {
    d_data[0] = size * size;
  }
}

int main(int argc, char const* argv[])
{
  size_t size = 42;
  size_t testing = 100;
  std::string default_dir = "./";

  std::stringstream ss;
  ss << default_dir << "umpire_mem_" << getpid() << "_0";

  int fd{open(ss.str().c_str(), O_RDWR | O_CREAT | O_LARGEFILE, S_IRWXU)};
  assert(fd != -1);

  const std::size_t pagesize{(std::size_t)sysconf(_SC_PAGE_SIZE)};
  std::size_t rounded_bytes{((size + (pagesize - 1)) / pagesize) * pagesize};

  int trun{ftruncate64(fd, rounded_bytes)};
  assert(trun != -1);

  size_t* ptr{
      static_cast<size_t*>(mmap(NULL, rounded_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0))};
  assert(ptr != MAP_FAILED);

  close(fd);
  
  ptr[0] = testing;
  std::cout << "ptr is: " << ptr[0] << " before" << std::endl;

  tester<<<1, 16>>>(ptr, size);
  cudaDeviceSynchronize();

  std::cout << "ptr is: " << ptr[0] << " after..." << std::endl;

  int rc = munmap(ptr, size);
  assert(rc == 0);
  
  return 0;
}
