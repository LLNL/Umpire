//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/util/MemoryResourceTraits.hpp"

TEST(MemoryResourceTraits, to_string_optimized_for)
{
  using namespace umpire;

  EXPECT_EQ("any", to_string(MemoryResourceTraits::optimized_for::any));
  EXPECT_EQ("latency", to_string(MemoryResourceTraits::optimized_for::latency));
  EXPECT_EQ("bandwidth", to_string(MemoryResourceTraits::optimized_for::bandwidth));
  EXPECT_EQ("access", to_string(MemoryResourceTraits::optimized_for::access));
}

TEST(MemoryResourceTraits, to_string_vendor_type)
{
  using namespace umpire;

  EXPECT_EQ("unknown", to_string(MemoryResourceTraits::vendor_type::unknown));
  EXPECT_EQ("amd", to_string(MemoryResourceTraits::vendor_type::amd));
  EXPECT_EQ("ibm", to_string(MemoryResourceTraits::vendor_type::ibm));
  EXPECT_EQ("intel", to_string(MemoryResourceTraits::vendor_type::intel));
  EXPECT_EQ("nvidia", to_string(MemoryResourceTraits::vendor_type::nvidia));
}

TEST(MemoryResourceTraits, to_string_memory_type)
{
  using namespace umpire;

  EXPECT_EQ("unknown", to_string(MemoryResourceTraits::memory_type::unknown));
  EXPECT_EQ("ddr", to_string(MemoryResourceTraits::memory_type::ddr));
  EXPECT_EQ("gddr", to_string(MemoryResourceTraits::memory_type::gddr));
  EXPECT_EQ("hbm", to_string(MemoryResourceTraits::memory_type::hbm));
  EXPECT_EQ("nvme", to_string(MemoryResourceTraits::memory_type::nvme));
}

TEST(MemoryResourceTraits, to_string_resource_type)
{
  using namespace umpire;

  EXPECT_EQ("unknown", to_string(MemoryResourceTraits::resource_type::unknown));
  EXPECT_EQ("host", to_string(MemoryResourceTraits::resource_type::host));
  EXPECT_EQ("device", to_string(MemoryResourceTraits::resource_type::device));
  EXPECT_EQ("device_const", to_string(MemoryResourceTraits::resource_type::device_const));
  EXPECT_EQ("pinned", to_string(MemoryResourceTraits::resource_type::pinned));
  EXPECT_EQ("um", to_string(MemoryResourceTraits::resource_type::um));
  EXPECT_EQ("file", to_string(MemoryResourceTraits::resource_type::file));
  EXPECT_EQ("shared", to_string(MemoryResourceTraits::resource_type::shared));
}

TEST(MemoryResourceTraits, to_string_granularity_type)
{
  using namespace umpire;

  EXPECT_EQ("unknown", to_string(MemoryResourceTraits::granularity_type::unknown));
  EXPECT_EQ("fine_grained", to_string(MemoryResourceTraits::granularity_type::fine_grained));
  EXPECT_EQ("coarse_grained", to_string(MemoryResourceTraits::granularity_type::coarse_grained));
}

TEST(MemoryResourceTraits, to_string_shared_scope)
{
  using namespace umpire;

  // Test the existing shared_scope to_string as well
  EXPECT_EQ("unknown", to_string(MemoryResourceTraits::shared_scope::unknown));
  EXPECT_EQ("node", to_string(MemoryResourceTraits::shared_scope::node));
  EXPECT_EQ("socket", to_string(MemoryResourceTraits::shared_scope::socket));
}
