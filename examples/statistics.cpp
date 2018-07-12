//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/util/StatisticsDatabase.hpp"
#include "umpire/util/Macros.hpp"

#include <iostream>

int main() {
  auto &rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");

  const int size = 100;
  void** allocs = new void*[size];

  for (int i = 0; i < size; i++) {
    allocs[i] = static_cast<double*>(alloc.allocate(100 * sizeof(double)));
  }

  for (int i = 0; i < size/2; i++) {
    rm.copy(allocs[i], allocs[size-1-i]);
  }

  for (int i = 0; i < size; i++) {
    alloc.deallocate(allocs[i]);
  }

  umpire::util::StatisticsDatabase::getDatabase()->printStatistics(std::cout);
}
