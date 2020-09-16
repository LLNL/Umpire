//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolMap.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

#include "umpire/Umpire.cpp"
#include "umpire/config.hpp"
#include "umpire/util/Platform.hpp"

using myResource = umpire::MemoryResourceTraits::resource_type;
using cPlatform = camp::resources::Platform;

void testAccess(umpire::Allocator a)
{
  std::cout << "Testing the host platform..." << std::endl;
  if(is_accessible(cPlatform::host, a)) {
    std::cout << "The allocator, " << a.getName() << 
                 ", is accessible." << std::endl << std::endl;
  } else {
    std::cout << "The allocator is not accessible."
              << std::endl << std::endl;
  }
  
  std::cout << "Testing the cuda platform..." << std::endl;
  if(is_accessible(cPlatform::cuda, a)) {
    std::cout << "The allocator, " << a.getName() << 
                 ", is accessible." << std::endl << std::endl;
  } else {
    std::cout << "The allocator is not accessible."
              << std::endl << std::endl;
  }
 
  std::cout << "Testing the hip platform..." << std::endl;
  if(is_accessible(cPlatform::hip, a)) {
    std::cout << "The allocator, " << a.getName() << 
                 ", is accessible." << std::endl << std::endl;
  } else {
    std::cout << "The allocator is not accessible."
              << std::endl << std::endl;
  }

  std::cout << "Testing the omp_target platform..." << std::endl;
  if(is_accessible(cPlatform::omp_target, a)) {
    std::cout << "The allocator, " << a.getName() << 
                 ", is accessible." << std::endl << std::endl;
  } else {
    std::cout << "The allocator is not accessible."
              << std::endl << std::endl;
  } 

  std::cout << "Testing the sycl platform..." << std::endl;
  if(is_accessible(cPlatform::sycl, a)) {
    std::cout << "The allocator, " << a.getName() << 
                 ", is accessible." << std::endl << std::endl;
  } else {
    std::cout << "The allocator is not accessible."
              << std::endl << std::endl;
  }
  
  std::cout << "---------------------------------" << std::endl; 
  return; 
}

int main()
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_alloc = rm.makeAllocator<umpire::strategy::DynamicPoolMap>(
      "host_DynamicPoolMap", rm.getAllocator("HOST"));

  auto dev_alloc = rm.makeAllocator<umpire::strategy::QuickPool>(
      "dev_QuickPool", rm.getAllocator("DEVICE"));
  
  double* dataH = static_cast<double*>(host_alloc.allocate(1024*sizeof(double)));
  testAccess(host_alloc);

  double* dataD = static_cast<double*>(dev_alloc.allocate(1024*sizeof(double)));
  testAccess(dev_alloc);

  host_alloc.deallocate(dataH);
  dev_alloc.deallocate(dataD);

  return 0;
}

