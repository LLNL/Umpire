//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC
// and Umpire project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <limits>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/device_memset_kernel.hpp"

int main()
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto dev_alloc = rm.getAllocator("DEVICE");
  constexpr std::size_t n = 1024;

  // Allocate DEVICE memory for n doubles.
  double* d_ptr = static_cast<double*>(dev_alloc.allocate(n * sizeof(double)));

  // Fill the DEVICE allocation with a value using the deviceMemset helper.
  const double value = 0.0;
  umpire::device_memset(d_ptr, n, value);

  // Allocate HOST memory to verify the contents with a copy back to host.
  auto host_alloc = rm.getAllocator("HOST");
  double* h_ptr = static_cast<double*>(host_alloc.allocate(n * sizeof(double)));

  rm.copy(h_ptr, d_ptr, n * sizeof(double));
  for (std::size_t i = 0; i < n; ++i) {
    if (h_ptr[i] != value) {
      std::cout << "DeviceMemset example FAILED!" << std::endl;
      break;
    }
  }

  std::cout << "DeviceMemset to " << value << " SUCCEEDED!" << std::endl;

  host_alloc.deallocate(h_ptr);
  dev_alloc.deallocate(d_ptr);

  return 0;
}

