//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/Umpire.hpp"
#include "umpire/op.hpp"
#include "umpire/resource/host_memory.hpp"

#include <cstddef>
#include <iostream>
#include <vector>

#if defined(UMPIRE_ENABLE_CUDA)
#include "umpire/resource/cuda_device_memory.hpp"
#include <cuda_runtime_api.h>
#endif

int main()
{
  using host_memory = umpire::resource::host_memory<>;

  auto& host = host_memory::get();
  std::vector<int> source{1, 4, 9, 16};
  auto* host_buffer = static_cast<int*>(host.allocate(source.size() * sizeof(int)));

  umpire::copy<umpire::host_platform, umpire::host_platform>(
    source.data(), host_buffer, source.size());
  umpire::memset<umpire::host_platform>(host_buffer + 3, 0, std::size_t{1});

  std::cout << "host copy result:";
  for (std::size_t i = 0; i < source.size(); ++i) {
    std::cout << ' ' << host_buffer[i];
  }
  std::cout << '\n';

#if defined(UMPIRE_ENABLE_CUDA)
  auto& device = umpire::resource::cuda_device_memory<>::get();
  auto* device_buffer = static_cast<int*>(device.allocate(source.size() * sizeof(int)));

  umpire::copy<umpire::host_platform, umpire::cuda_platform>(
    source.data(), device_buffer, source.size());
  umpire::copy<umpire::cuda_platform, umpire::host_platform>(
    device_buffer, host_buffer, source.size());

  std::cout << "cuda round-trip result:";
  for (std::size_t i = 0; i < source.size(); ++i) {
    std::cout << ' ' << host_buffer[i];
  }
  std::cout << '\n';

  device.deallocate(device_buffer);
#else
  std::cout << "CUDA backend disabled; host-to-device copy example is code-only on this machine.\n";
#endif

  host.deallocate(host_buffer);
  return 0;
}
