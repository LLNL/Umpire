//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/util/error.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("UM");

  // _sphinx_tag_tut_device_advice_start
  //
  // Create an allocator that applied "PREFFERED_LOCATION" advice to set a
  // specific GPU device as the preferred location.
  //
  // In this case, device #2.
  //
  const int device_id = 2;

  try {
    auto preferred_location_allocator = rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
        "preferred_location_device_2", allocator, "SET_PREFERRED_LOCATION", device_id);

    // _sphinx_tag_tut_device_advice_end
    void* data = preferred_location_allocator.allocate(1024);

    preferred_location_allocator.deallocate(data);
  } catch (umpire::runtime_error& e) {
    std::cout << "Couldn't create Allocator with device_id = " << device_id << std::endl;
    std::cout << e.message() << std::endl;
  }

  return 0;
}
