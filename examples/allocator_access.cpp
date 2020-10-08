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
#include "umpire/Umpire.hpp"

using cPlatform = camp::resources::Platform;

bool testAccess(umpire::Allocator a)
{
  if(umpire::is_accessible(cPlatform::host, a)) {
    std::cout << "The allocator, " << a.getName()
              << ", is accessible." << std::endl;
    return true;
  } else {
    std::cout << "The allocator, " << a.getName() 
              << ", is _not_ accessible." << std::endl << std::endl;
    return false;
  }
}

///////////////////////////////////////////////////
//Depending on how Umpire has been set up, several
//different allocators could be accessible from the host
//CAMP platform. This test will create a list of all
//currently available allocators and then test each
//individually to see if it can be accessed from the
//*host* platform. (To test other platforms, see test.)
///////////////////////////////////////////////////
int main()
{
  auto& rm = umpire::ResourceManager::getInstance();

  std::vector<std::string> allNames = rm.getResourceNames();
  std::vector<umpire::Allocator> alloc;
  
  ///////////////////////////////////////////////////
  //Create an allocator for each available type
  //////////////////////////////////////////////////
  std::cout << "Available allocators: ";
  for(auto a : allNames) {
    if (a.find("::") == std::string::npos) { 
      alloc.push_back(rm.getAllocator(a));
      std::cout << a << " ";
    }
  }
  std::cout<<std::endl<<std::endl;

  ///////////////////////////////////////////////////
  //Test accessibility
  ///////////////////////////////////////////////////
  std::cout << "Testing the available allocators for "
            << "accessibility from the CAMP host platform:" 
            << std::endl << std::endl;
  const int size = 100;
  for(auto a : alloc) {
    if(testAccess(a)) {
      int* data = static_cast<int*>(a.allocate(size*sizeof(int)));
      for(int i = 0; i < size; i++) {
        data[i] = i * i;
      }
      std::cout << "Testing: " << data[size-1] << " should be equal to " 
                << (size-1)*(size-1) << std::endl << std::endl; 
    }
  }
 
  return 0;
}

