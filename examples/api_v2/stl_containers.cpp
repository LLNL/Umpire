//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/allocator.hpp"
#include "umpire/resource/host_memory.hpp"

#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

int main()
{
  using host_memory = umpire::resource::host_memory<>;
  auto& host = host_memory::get();

  umpire::allocator<int, host_memory> int_alloc{&host};
  std::vector<int, umpire::allocator<int, host_memory>> values{int_alloc};

  values.reserve(8);
  values.push_back(3);
  values.push_back(5);
  values.resize(4);
  values[2] = 8;
  values[3] = 13;

  using pair_type = std::pair<const int, std::string>;
  std::map<int, std::string, std::less<int>, umpire::allocator<pair_type, host_memory>> labels{
    umpire::allocator<pair_type, host_memory>{&host}};
  labels.emplace(3, "three");
  labels.emplace(13, "thirteen");

  std::deque<int, umpire::allocator<int, host_memory>> window{int_alloc};
  window.push_back(values.front());
  window.push_back(values.back());

  auto shared = std::allocate_shared<std::string>(
    umpire::allocator<std::string, host_memory>{&host},
    "api_v2 host allocator");

  std::cout << "vector:";
  for (int value : values) {
    std::cout << ' ' << value;
  }
  std::cout << "\nmap[13]: " << labels.at(13)
            << "\ndeque back: " << window.back()
            << "\nshared: " << *shared << '\n';

  return 0;
}
