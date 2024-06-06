//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/AlignedAllocator.hpp"

int main()
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto aligned_alloc =
      rm.makeAllocator<umpire::strategy::AlignedAllocator>("aligned_allocator", rm.getAllocator("HOST"), 256);

  void* data = aligned_alloc.allocate(1234);
  aligned_alloc.deallocate(data);

  data = aligned_alloc.allocate(7);
  aligned_alloc.deallocate(data);

  data = aligned_alloc.allocate(5555);
  aligned_alloc.deallocate(data);

  return 0;
}
