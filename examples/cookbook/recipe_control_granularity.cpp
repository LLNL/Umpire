//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include <vector>
#include <hip/hip_runtime_api.h>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/GranularityController.hpp"
// #include "umpire/strategy/QuickPool.hpp"

// Statistics information for allocated memory
struct myMemStats {
    friend std::ostream& operator<< (std::ostream& stream, const myMemStats& mstats);

    myMemStats(void* ptr) : name{umpire::ResourceManager::getInstance().getAllocator(ptr).getName()}
    {
        if (::hipPointerGetAttributes(&hipattrs, ptr) != hipSuccess) {
          std::cout << "Allocator: (" << name << ") Error: hipPointerGetAttributes failed for address: " << ptr << std::endl;
        }
    }

    void print(std::ostream& stream) const {
        stream << name << std::endl
          << "        Memory Type: " << hipattrs.memoryType << std::endl
          << "             device: " << hipattrs.device << std::endl
          << "        hostPointer: " << hipattrs.hostPointer << std::endl
          << "      devicePointer: " << hipattrs.devicePointer << std::endl
          << "    allocationFlags: " << hipattrs.allocationFlags << std::endl;
    }

    hipPointerAttribute_t hipattrs;
    std::string name;
};

std::ostream& operator<< (std::ostream& stream, const myMemStats& mstats) {
    mstats.print(stream);
    return stream;
}


int main(int, char**)
{
  const std::vector<std::string> resources{"DEVICE", "UM", "PINNED"};

  for ( auto&& resource : resources ) {
    const std::vector<std::pair<umpire::strategy::GranularityController::Granularity, std::string>> mtypes{
      { umpire::strategy::GranularityController::Granularity::CoarseGrainedCoherence, resource + "_COURSE" },
      { umpire::strategy::GranularityController::Granularity::FineGrainedCoherence, resource + "_FINE" }
    };
    auto& rm = umpire::ResourceManager::getInstance();
    auto allocator = rm.getAllocator(resource);

    for (auto&& mtype : mtypes) {
      auto alloc = rm.makeAllocator<umpire::strategy::GranularityController>(
        mtype.second, 
        allocator, 
        mtype.first);

      std::vector<void*> ptrs;
      const int N{1};
      int size{2};

      for ( int i = 0; i < N; i++) {
        ptrs.push_back(alloc.allocate(size));
        size *= 2;
      }

      for ( int i = 0; i < N; i++) {
        myMemStats stat{ptrs[i]};
        std::cout << stat << std::endl;
        alloc.deallocate(ptrs[i]);
      }
    }
  }

  return 0;
}
