//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/TypedAllocator.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");

  // _umpire_tut_typed_alloc_start
  umpire::TypedAllocator<double> double_allocator{alloc};

  double* my_doubles = double_allocator.allocate(1024);

  double_allocator.deallocate(my_doubles);
  // _umpire_tut_typed_alloc_end

  // _umpire_tut_vector_alloc_start
  std::vector<double, umpire::TypedAllocator<double>> my_vector{
      double_allocator};
  // _umpire_tut_vector_alloc_end

  my_vector.resize(100);

  return 0;
}
