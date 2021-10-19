//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/Umpire.hpp"

TEST(Umpire, ProcessorMemoryStatistics)
{
  ASSERT_GE(umpire::get_process_memory_usage(), 0);
  ASSERT_GE(umpire::get_process_memory_usage_hwm(), umpire::get_process_memory_usage());
  ASSERT_GE(umpire::get_device_memory_usage(0), 0);
}
