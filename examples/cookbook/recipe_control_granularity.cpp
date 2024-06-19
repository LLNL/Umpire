//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <hip/hip_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

// Statistics information for allocated memory
struct myMemStats {
  friend std::ostream& operator<<(std::ostream& stream, const myMemStats& mstats);

  myMemStats(void* ptr) : name{umpire::ResourceManager::getInstance().getAllocator(ptr).getName()}
  {
    if (::hipPointerGetAttributes(&hipattrs, ptr) != hipSuccess) {
      std::cout << "Allocator: (" << name << ") Error: hipPointerGetAttributes failed for address: " << ptr
                << std::endl;
    }
  }

  void print(std::ostream& stream) const
  {
    stream << name << std::endl
           << "        Memory Type: " << hipattrs.type << std::endl
           << "             device: " << hipattrs.device << std::endl
           << "        hostPointer: " << hipattrs.hostPointer << std::endl
           << "      devicePointer: " << hipattrs.devicePointer << std::endl
           << "    allocationFlags: " << hipattrs.allocationFlags << std::endl;
  }

  hipPointerAttribute_t hipattrs;
  std::string name;
};

std::ostream& operator<<(std::ostream& stream, const myMemStats& mstats)
{
  mstats.print(stream);
  return stream;
}

int main(int, char**)
{
  const std::vector<std::string> resources{"DEVICE::COARSE", "DEVICE::FINE", "DEVICE::0::COARSE", "UM::FINE",
                                           "UM::COARSE",     "PINNED::FINE", "PINNED::COARSE"};

  for (auto&& resource : resources) {
    auto& rm = umpire::ResourceManager::getInstance();
    auto alloc = rm.getAllocator(resource);

    std::vector<void*> ptrs;
    const int N{1};
    int size{2};

    for (int i = 0; i < N; i++) {
      ptrs.push_back(alloc.allocate(size));
      size *= 2;
    }

    for (int i = 0; i < N; i++) {
      myMemStats stat{ptrs[i]};
      std::cout << stat << std::endl;
      alloc.deallocate(ptrs[i]);
    }
  }

  return 0;
}
