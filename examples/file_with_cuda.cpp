//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

#include "umpire/resource/FileMemoryResource.hpp"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

__global__ void tester(size_t* d_data, size_t size)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
   
  if (idx == 0) {
    d_data[0] = size * size;
  }
}

int main(int argc, char const* argv[])
{
  auto& rm = umpire::ResourceManager::getInstance();
  //auto allocator = rm.getAllocator("FILE");

  size_t size = 42;
  size_t testing = 100;

 const char* memory_file_dir{std::getenv("UMPIRE_MEMORY_FILE_DIR")};
  std::string default_dir = "./";
  if (memory_file_dir) {
    default_dir = memory_file_dir;
  }

std::stringstream ss;
  ss << default_dir << "umpire_mem_" << getpid() << "_0";

  int fd{open(ss.str().c_str(), O_RDWR | O_CREAT | O_LARGEFILE, S_IRWXU)};
  if (fd == -1) {
    UMPIRE_ERROR("Opening File { " << ss.str()
                                   << " } Failed: " << strerror(errno));
  }

const std::size_t pagesize{(std::size_t)sysconf(_SC_PAGE_SIZE)};
  std::size_t rounded_bytes{((size + (pagesize - 1)) / pagesize) * pagesize};

  int trun{ftruncate64(fd, rounded_bytes)};
  if (trun == -1) {
    int errno_save = errno;
    remove(ss.str().c_str());
    UMPIRE_ERROR("truncate64 Of File { "
                 << ss.str() << " } Failed: " << strerror(errno_save));
  }

size_t* ptr{
      static_cast<size_t*>(mmap(NULL, rounded_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0))};
  if (ptr == MAP_FAILED) {
    int errno_save = errno;
    remove(ss.str().c_str());
    UMPIRE_ERROR("mmap Of " << rounded_bytes << " To File { " << ss.str()
                            << " } Failed: " << strerror(errno_save));
  }

 close(fd);
  ptr[0] = testing;
  std::cout << "ptr is: " << ptr[0] << " before" << std::endl;

  tester<<<1, 16>>>(ptr, size);
  cudaDeviceSynchronize();

  std::cout << "ptr is: " << ptr[0] << " after..." << std::endl;

  //allocator.deallocate(dataB);
  //allocator.deallocate(dataA);
  return 0;
}
