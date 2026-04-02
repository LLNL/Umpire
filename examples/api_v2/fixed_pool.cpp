//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/host_memory.hpp"
#include "umpire/strategy/fixed_pool.hpp"

#include <array>
#include <iostream>

int main()
{
  using host_memory = umpire::resource::host_memory<>;
  using pool_type = umpire::strategy::fixed_pool<host_memory>;

  auto& host = host_memory::get();
  pool_type pool{"PARTICLE_POOL", &host, sizeof(double), 8};

  std::array<double*, 4> values{};
  for (auto& value : values) {
    value = static_cast<double*>(pool.allocate(sizeof(double)));
  }

  for (std::size_t i = 0; i < values.size(); ++i) {
    *values[i] = static_cast<double>(i + 1) * 1.5;
  }

  std::cout << "pool values:";
  for (double* value : values) {
    std::cout << ' ' << *value;
    pool.deallocate(value);
  }
  std::cout << "\npools before release: " << pool.get_pool_count() << '\n';

  pool.release();
  std::cout << "pools after release: " << pool.get_pool_count() << '\n';
  return 0;
}
