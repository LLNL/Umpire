//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/util/Exception.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("UM");

  /*
   * Create an allocator that applied "PREFFERED_LOCATION" advice to set a
   * specific GPU device as the preferred location.
   *
   * In this case, device #2.
   */
  const int device_id = 2;

  try {
    auto preferred_location_allocator =
        rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
            "preferred_location_device_2", allocator, "PREFERRED_LOCATION",
            device_id);

    void* data = preferred_location_allocator.allocate(1024);

    preferred_location_allocator.deallocate(data);
  } catch (umpire::util::Exception& e) {
    std::cout << "Couldn't create Allocator with device_id = " << device_id
              << std::endl;

    std::cout << e.message() << std::endl;
  }

  return 0;
}
