//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/StatisticsDatabase.hpp"

int main()
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");

  const int size = 100;
  void** allocs = new void*[size];

  for (int i = 0; i < size; i++) {
    allocs[i] = static_cast<double*>(alloc.allocate(100 * sizeof(double)));
  }

  for (int i = 0; i < size / 2; i++) {
    rm.copy(allocs[i], allocs[size - 1 - i]);
  }

  for (int i = 0; i < size; i++) {
    alloc.deallocate(allocs[i]);
  }

  umpire::util::StatisticsDatabase::getDatabase()->printStatistics(std::cout);
}
