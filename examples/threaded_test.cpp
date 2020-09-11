//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/AlignedAllocator.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"

int main()
{
  auto& rm = umpire::ResourceManager::getInstance();
  //auto aligned_alloc = rm.makeAllocator<umpire::strategy::AlignedAllocator>(
  //    "aligned_allocator", rm.getAllocator("HOST"), 256);

  //auto dynamic_alloc = rm.makeAllocator<umpire::strategy::DynamicPoolList>(
  //    "dynamic_allocator", rm.getAllocator("HOST"));
  
  auto threaded_alloc = rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      "threaded_allocator", rm.getAllocator("HOST"));
  
  //void* data = aligned_alloc.allocate(1234);
  //if(aligned_alloc.getThreaded())
  //  std::cout << "Aligned Alloc is threaded!" << std::endl; 
  //aligned_alloc.deallocate(data);

  //data = dynamic_alloc.allocate(1234);
  //if(dynamic_alloc.getThreaded())
  //  std::cout << "DynamicPoolList Alloc is threaded!" << std::endl; 
  //dynamic_alloc.deallocate(data);
  
  void* data = threaded_alloc.allocate(1234);
  if(threaded_alloc.getThreaded() == true)
    std::cout << "ThreadSafe Alloc is threaded!" << std::endl; 
  else if (threaded_alloc.getThreaded() == false)
    std::cout << "Error!" << std::endl;
  else
    std::cout << "A real problematic error!" << std::endl;

  threaded_alloc.deallocate(data);
  
  return 0;
}
