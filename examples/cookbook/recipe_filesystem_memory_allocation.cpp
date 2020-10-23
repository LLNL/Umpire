//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

int main(int, char** argv)
{
  // _umpire_tut_file_allocate_start
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("FILE");

  std::size_t* A = (std::size_t*)alloc.allocate(sizeof(size_t));
  // _umpire_tut_file_allocate_end

  // _umpire_tut_file_deallocate_start
  alloc.deallocate(A);
  // _umpire_tut_file_deallocate_end

  return 0;
}
