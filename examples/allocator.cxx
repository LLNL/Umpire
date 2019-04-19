//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#include <iostream>

int main() {
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("HOST");

  std::cout << "Got allocator: " << alloc.getName() << std::endl;

  std::cout << "Available allocators: ";
  for (auto s : rm.getAllocatorNames()){
    std::cout << s << "  ";
  }
  std::cout << std::endl;

  return 0;
}
