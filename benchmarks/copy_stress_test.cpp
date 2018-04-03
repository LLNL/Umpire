//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#include <chrono>

#define ALLOCATIONS 100000

int do_copy(std::string src, std::string dst, std::size_t size = 4096) {
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator src_alloc = rm.getAllocator(src);
  umpire::Allocator dst_alloc = rm.getAllocator(dst);

  void* src_allocations[ALLOCATIONS];
  void* dst_allocations[ALLOCATIONS];

  for (int i = 0; i < ALLOCATIONS; i++) {
    src_allocations[i] = src_alloc.allocate(size);
    dst_allocations[i] = dst_alloc.allocate(size);
  }

  auto begin_copy = std::chrono::system_clock::now();
  for (int i = 0; i < ALLOCATIONS; i++) {
    rm.copy(src_allocations[i], dst_allocations[i]);
  }
  auto end_copy = std::chrono::system_clock::now();

  for (int i = 0; i < ALLOCATIONS; i++) {
    src_alloc.deallocate(src_allocations[i]);
    dst_alloc.deallocate(dst_allocations[i]);
  }

  std::cout << src << "->" << dst << std::endl;
  std::cout << "    copy: " <<  std::chrono::duration<double>(end_copy - begin_copy).count()/ALLOCATIONS << std::endl;
  
  return 0;
}

int main(int, char**) {
  do_copy("HOST", "HOST");

#if defined(UMPIRE_ENABLE_CUDA)
  do_copy("HOST", "DEVICE");
  do_copy("DEVICE", "HOST");
  do_copy("DEVICE", "DEVICE");

  do_copy("HOST", "UM");
  do_copy("UM", "HOST");
  do_copy("UM", "UM");

  do_copy("UM", "DEVICE");
  do_copy("DEVICE", "UM");
#endif
}
