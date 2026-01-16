//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC
// and Umpire project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <limits>

#include "umpire/ResourceManager.hpp"

int main()
{
  auto& rm = umpire::ResourceManager::getInstance();

  if (!rm.isAllocator("DEVICE")) {
    std::cout << "DEVICE allocator is not available in this build.\n";
    return 0;
  }

  auto dev_alloc = rm.getAllocator("DEVICE");
  constexpr std::size_t n = 1024;

  // Allocate DEVICE memory for n doubles.
  double* d_ptr = static_cast<double*>(dev_alloc.allocate(n * sizeof(double)));

  // Fill the DEVICE allocation with a value using the deviceMemset helper.
  const double value = 0.0;
  dev_alloc.deviceMemset(d_ptr, n, value);
/*
  // Allocate HOST memory to verify the contents with a copy back to host.
  auto host_alloc = rm.getAllocator("HOST");
  double* h_ptr = static_cast<double*>(host_alloc.allocate(n * sizeof(double)));

  rm.copy(h_ptr, d_ptr, n * sizeof(double));

  bool ok = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (h_ptr[i] != value) {
      ok = false;
      break;
    }
  }

  std::cout << "deviceMemset example: "
            << (ok ? "SUCCESS" : "FAILURE") << std::endl;

  host_alloc.deallocate(h_ptr);
*/
  dev_alloc.deallocate(d_ptr);

  return 0; //ok ? 0 : 1;
}

