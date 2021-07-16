//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/allocator.hpp"
#include "umpire/umpire.hpp"

#include <iostream>

int main(int, char**) {
  constexpr std::size_t SIZE = 1024;

  umpire::initialize();

  auto strategy = umpire::get_strategy("HOST");
  umpire::allocator<double>  allocator{strategy};

  double* data = allocator.allocate(SIZE);
  std::cout << "Allocated " << SIZE << typeid(umpire::allocator<double>::value_type).name() << " using the " << allocator.get_name() << " allocator...";

  allocator.deallocate(data);

  std::cout << " deallocated." << std::endl;

  return 0;
}
