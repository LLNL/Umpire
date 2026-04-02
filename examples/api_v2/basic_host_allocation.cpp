//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/allocator.hpp"
#include "umpire/resource/host_memory.hpp"

#include <iostream>
#include <vector>

int main()
{
  using host_memory = umpire::resource::host_memory<>;

  auto& host = host_memory::get();
  void* raw = host.allocate(256);
  host.deallocate(raw);

  umpire::allocator<double, host_memory> alloc{&host};
  std::vector<double, umpire::allocator<double, host_memory>> values{alloc};

  values.reserve(4);
  values.push_back(1.5);
  values.push_back(2.5);
  values.push_back(4.0);
  values.push_back(8.0);

  std::cout << "vector size: " << values.size()
            << "\nhost allocator id: " << alloc.get_id()
            << "\nlast value: " << values.back() << '\n';

  return 0;
}
