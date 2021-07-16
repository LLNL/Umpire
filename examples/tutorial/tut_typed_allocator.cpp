//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/umpire.hpp"

namespace umpire {
  template <typename T>
  using vector = std::vector<T, allocator<T> >;
}

int main(int, char**) {
  umpire::initialize();

  umpire::allocator<double> allocator{umpire::get_strategy("HOST")};
  umpire::vector my_vector{double_allocator};
  my_vector.resize(100);

  return 0;
}
