//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include "umpire/util/Macros.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolMap.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

#include "umpire/Umpire.cpp"
#include "umpire/config.hpp"
#include "umpire/util/Platform.hpp"

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

  ///////////////////////////////////////////////////
  //Create an allocator for each Memory Resource Type
  //(For fun, some allocators are tested on different
  //strategies.)

  auto host_alloc = rm.makeAllocator<umpire::strategy::DynamicPoolMap>(
      "host_DynamicPoolMap", rm.getAllocator("HOST"));

  auto dev_alloc = rm.makeAllocator<umpire::strategy::QuickPool>(
      "dev_QuickPool", rm.getAllocator("DEVICE"));
  
  auto file_alloc = rm.getAllocator("FILE");
  
 // auto const_alloc = rm.getAllocator("DEVICE_CONST");
  
  //auto pin_alloc = rm.makeAllocator<umpire::strategy::SizeLimiter>(
    //  "pin_SizeLimiter", rm.getAllocator("PINNED"), 1024*sizeof(double));
  
  //auto um_alloc = rm.makeAllocator<umpire::strategy::DynamicPoolList>(
    //  "um_DynamicPoolList", rm.getAllocator("UM"));
  
  //////////////////////////////////////////////////////////////////
  //Allocate memory on each of the allocators and test accessibility
//TODO use new ResourceManager function to get list of available allocators
//and then loop though them

  std::vector<std::string> allNames = rm.getResourceNames();
  
  std::cout << "Names: ";
  for (auto s : allNames) {
    std::cout << s << " ";
  }
 
  double* dataH = static_cast<double*>(host_alloc.allocate(1024*sizeof(double)));
  std::cout << "HOST" << std::endl;
  testAccess(host_alloc);

  double* dataD = static_cast<double*>(dev_alloc.allocate(1024*sizeof(double)));
  std::cout << "DEVICE" << std::endl;
  testAccess(dev_alloc);
  
  int* dataF = static_cast<int*>(file_alloc.allocate(1024));
  std::cout << "FILE" << std::endl;
  testAccess(file_alloc);
  
  //int* dataC = static_cast<int*>(const_alloc.allocate(1024));
  //std::cout << "DEVICE_CONST" << std::endl;
  //testAccess(const_alloc);
  
  //double* dataP = static_cast<double*>(pin_alloc.allocate(1024*sizeof(double)));
  //std::cout << "PINNED" << std::endl;
  //testAccess(pin_alloc);

  //double* dataU = static_cast<double*>(um_alloc.allocate(1024*sizeof(double)));
  //std::cout << "UM" << std::endl;
  //testAccess(um_alloc);
  
  /////////////////////////////
  //Clean up, deallocate memory

  host_alloc.deallocate(dataH);
  dev_alloc.deallocate(dataD);
  file_alloc.deallocate(dataF);
  //const_alloc.deallocate(dataC);
  //pin_alloc.deallocate(dataP);
  //um_alloc.deallocate(dataU);

  return 0;
}

