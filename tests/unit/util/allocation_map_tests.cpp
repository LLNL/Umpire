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
#include "umpire/util/AllocationMap.hpp"

#include "umpire/util/AllocationRecord.hpp"

#include "umpire/util/Exception.hpp"

#include "gtest/gtest.h"

TEST(AllocationMap, AddFindRemove)
{
  umpire::util::AllocationMap map;

  int* pointer = new int[10];

  umpire::util::AllocationRecord* record = new umpire::util::AllocationRecord{pointer, 10, nullptr};

  map.insert((void*)pointer, record);

  auto found_record = map.find(pointer);

  ASSERT_EQ(record, found_record);

  map.remove(pointer);

  ASSERT_THROW(
      map.find(pointer),
      umpire::util::Exception);
}
