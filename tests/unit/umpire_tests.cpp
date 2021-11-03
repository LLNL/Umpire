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

  //
  // Be careful with the following test.  The _hwm call must be called last when compared
  // to the live amount of system memory in use
  //
  ASSERT_LE(umpire::get_process_memory_usage(), umpire::get_process_memory_usage_hwm());

  ASSERT_GE(umpire::get_device_memory_usage(0), 0);
}
