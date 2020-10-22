//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

int main(int, char**)
{
  // _umpire_tut_get_allocator_start
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("HOST");
  // _umpire_tut_get_allocator_end

  // _umpire_tut_de_allocate_start
  constexpr std::size_t SIZE = 1024;

  double* data =
      static_cast<double*>(allocator.allocate(SIZE * sizeof(double)));

  std::cout << "Allocated " << (SIZE * sizeof(double)) << " bytes using the "
            << allocator.getName() << " allocator...";

  allocator.deallocate(data);
  // _umpire_tut_de_allocate_end

  std::cout << " deallocated." << std::endl;

  return 0;
}

