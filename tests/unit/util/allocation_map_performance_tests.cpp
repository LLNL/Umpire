//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#include "umpire/util/AllocationMap.hpp"

#include "umpire/util/AllocationRecord.hpp"

#include "umpire/util/Exception.hpp"

#include "gtest/gtest.h"

#include <chrono>

class AllocationMapPerformanceTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      num_entries = 10000;

      size = 8;

      data = new double*[num_entries];
      records = new umpire::util::AllocationRecord*[num_entries];

      for (size_t i = 0; i < num_entries; i++) {
        data[i] = new double[size];
        records[i] = new umpire::util::AllocationRecord{data[i], size, nullptr};
      }

    }

    virtual void TearDown() {
      for (size_t i = 0; i < num_entries; i++) {
        delete[] data[i];
        delete records[i];
      }

      delete[] data;
      delete[] records;
    }

    umpire::util::AllocationMap map;

    double** data;
    size_t size;
    size_t num_entries;
    umpire::util::AllocationRecord** records;
};

TEST_F(AllocationMapPerformanceTest, Insert)
{
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_entries; ++i) {
    map.insert(data[i], records[i]);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  // TODO: make this a better test
  ASSERT_LT(elapsed_ns.count()/num_entries, 1000);
}
