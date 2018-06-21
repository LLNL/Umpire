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

class AllocationMapTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      double* data = new double[15];
      size = 15;
      record = new umpire::util::AllocationRecord{data, size, nullptr};
    }

    virtual void TearDown() {
    }

    umpire::util::AllocationMap map;

    double* data;
    size_t size;
    umpire::util::AllocationRecord* record;
};

TEST_F(AllocationMapTest, Add)
{
  EXPECT_NO_THROW(
    map.insert(data, record)
  );
}

TEST_F(AllocationMapTest, FindNotFound)
{
  ASSERT_THROW(
    map.find(data),
    umpire::util::Exception);
}

TEST_F(AllocationMapTest, Find)
{
  EXPECT_NO_THROW(
    map.insert(data,record)
  );

  auto actual_record = map.find(data);

  ASSERT_EQ(record, actual_record);
}

TEST_F(AllocationMapTest, Contains)
{
  EXPECT_NO_THROW(
    map.insert(data,record)
  );

  ASSERT_TRUE(map.contains(data));
}

TEST_F(AllocationMapTest, NotContains)
{
  ASSERT_FALSE(map.contains(data));
}

TEST_F(AllocationMapTest, Remove)
{
  EXPECT_NO_THROW({
    map.insert(data,record);
    map.remove(data);
  });

  ASSERT_THROW(
      map.find(data),
      umpire::util::Exception);

  ASSERT_FALSE(map.contains(data));
}

TEST_F(AllocationMapTest, RemoveNotFound)
{
  ASSERT_THROW(
    map.remove(data),
    umpire::util::Exception
  );
}

TEST_F(AllocationMapTest, RegisterMultiple)
{
  umpire::util::AllocationRecord* next_record = 
    new umpire::util::AllocationRecord{data, 1, nullptr};

  ASSERT_NO_THROW(
    map.insert(data, record);
    map.insert(data, next_record);
  );
}

TEST_F(AllocationMapTest, FindMultiple)
{
  umpire::util::AllocationRecord* next_record = 
    new umpire::util::AllocationRecord{data, 1, nullptr};

  EXPECT_NO_THROW(
    map.insert(data, record);
    map.insert(data, next_record);
  );

  auto actual_record = map.find(data);

  ASSERT_EQ(next_record, actual_record);

  map.remove(data);

  EXPECT_NO_THROW(
    actual_record = map.find(data);
  );

  ASSERT_EQ(next_record, record);
}
