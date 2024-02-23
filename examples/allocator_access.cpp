//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"

bool is_accessible_from_host(umpire::Allocator a)
{
  if (umpire::is_accessible(umpire::Platform::host, a)) {
    std::cout << "The allocator, " << a.getName() << ", is accessible." << std::endl;
    return true;
  } else {
    std::cout << "The allocator, " << a.getName() << ", is _not_ accessible." << std::endl << std::endl;
    return false;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////
// Depending on how Umpire has been set up, several different allocators could be accessible
// from the host CAMP platform. This example will create a list of all currently available
// allocators and then determine whether each can be accessed from the host platform.
//(To test other platforms, see allocator accessibility test.)
////////////////////////////////////////////////////////////////////////////////////////
int main()
{
  auto& rm = umpire::ResourceManager::getInstance();

  std::vector<std::string> allNames = rm.getResourceNames();
  std::vector<umpire::Allocator> alloc;

  ///////////////////////////////////////////////////
  // Create an allocator for each available type
  //////////////////////////////////////////////////
  std::cout << "Available allocators: ";
  for (auto a : allNames) {
    if (a.find("::") == std::string::npos) {
      alloc.push_back(rm.getAllocator(a));
      std::cout << a << " ";
    }
  }
  std::cout << std::endl;

  ///////////////////////////////////////////////////
  // Test accessibility
  ///////////////////////////////////////////////////
  std::cout << "Testing the available allocators for accessibility from the CAMP host platform:" << std::endl;
  const int size = 100;
  for (auto a : alloc) {
    if (is_accessible_from_host(a)) {
      int* data = static_cast<int*>(a.allocate(size * sizeof(int)));
      for (int i = 0; i < size; i++) {
        data[i] = i * i;
      }
      UMPIRE_ASSERT(data[size - 1] == (size - 1) * (size - 1) && "Inequality found in array that should be accessible");
    }
  }

  return 0;
}
