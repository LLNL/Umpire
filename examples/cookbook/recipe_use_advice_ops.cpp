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
  auto set_advice_op = rm.getOperation("READ_MOSTLY", allocator, allocator);

  auto unset_advice_op =
      rm.getOperation("UNSET_READ_MOSTLY", allocator, allocator);

  constexpr size_t size = 1024 * sizeof(double);
  double* data = static_cast<double*>(allocator.allocate(size));

  //
  // Set the preferred location of data to the GPU
  //
  set_advice_op->apply(data,    // pointer to data
                       nullptr, // pointer record (UNUSED)
                       0,       // device id
                       size);   // size

  data[10] = 3.14;

  //
  // Unset the preferred location of data to the GPU
  //
  unset_advice_op->apply(data,    // pointer to data
                         nullptr, // pointer record (UNUSED)
                         0,       // device id
                         size);   // size

  allocator.deallocate(data);

  return 0;
}
