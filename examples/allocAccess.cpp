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
#include "umpire/Umpire.cpp"

//////////////////////////////////////////////////////////////////////////
//Some resource types may be repeated. If only the unique types should be
//tested, define this value. For example, if there is a DEVICE::1 and a 
//DEVICE resource type, only DEVICE will be tested for accessibility. 
//////////////////////////////////////////////////////////////////////////
#define UNIQUE

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

  std::vector<std::string> allNames = rm.getResourceNames();
  std::vector<umpire::Allocator> alloc;
  
  ///////////////////////////////////////////////////
  //Create an allocator for each available type
  for(auto a : allNames) {
    alloc.push_back(rm.getAllocator(a));
  }
  
  ///////////////////////////////////////////////////
  //Test accessibility
  for(int c = 0; c < alloc.size(); c++) {
#if defined UNIQUE
    if (allNames[c].find("::") == std::string::npos)
#endif
    {
      std::cout << allNames[c] << std::endl;
      testAccess(alloc[c]);
    }
  }
 
  return 0;
}

