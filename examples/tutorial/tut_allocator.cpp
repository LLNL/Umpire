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
  auto& rm = umpire::ResourceManager::getInstance();

  // _sphinx_tag_tut_get_allocator_start
  umpire::Allocator allocator = rm.getAllocator("HOST");
  // _sphinx_tag_tut_get_allocator_end

  constexpr std::size_t SIZE = 1024;

  // _sphinx_tag_tut_allocate_start
  double* data =
      static_cast<double*>(allocator.allocate(SIZE * sizeof(double)));
  // _sphinx_tag_tut_allocate_end

  std::cout << "Allocated " << (SIZE * sizeof(double)) << " bytes using the "
            << allocator.getName() << " allocator." << std::endl;

  //_sphinx_tag_tut_getAllocator_start
  auto addon_allocator = rm.getAllocator(allocator.getName());
  //_sphinx_tag_tut_getAllocator_end
  
  std::cout << "Created an add-on allocator using the " << addon_allocator.getName()
            << " allocator." << std::endl;

  // _sphinx_tag_tut_deallocate_start
  allocator.deallocate(data);
  // _sphinx_tag_tut_deallocate_end

  std::cout << "...Memory deallocated." << std::endl;

  return 0;
}

