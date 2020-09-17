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
  std::vector<std::string> allocAvail;
  std::vector<umpire::Allocator> alloc;
  
  ///////////////////////////////////////////////////
  //Create a list of unique available resource types
  for (auto s : allNames) {
    if (s.find("::") == std::string::npos){
      allocAvail.push_back(s);
    }
  }

  ///////////////////////////////////////////////////
  //Create an allocator for each available type
  for(auto a : allocAvail) {
    alloc.push_back(rm.getAllocator(a));
  }
  
  ///////////////////////////////////////////////////
  //Test accessibility
  int c = 0;
  for(auto a : alloc) {
    std::cout << allocAvail[c] << std::endl;
    testAccess(a);
    c++;
  }
 
  return 0;
}

