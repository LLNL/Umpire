//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/util/AllocationMap.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Exception.hpp"

// Define equality operators for tests
namespace umpire {
namespace util {

bool operator==(const umpire::util::AllocationRecord& left,
                const umpire::util::AllocationRecord& right)
{
  return left.ptr == right.ptr && left.size == right.size &&
         left.strategy == right.strategy;
}

bool operator!=(const umpire::util::AllocationRecord& left,
                const umpire::util::AllocationRecord& right)
{
  return !(left == right);
}

} // end namespace util
} // end namespace umpire

class AllocationMapTest : public ::testing::Test {
 protected:
  AllocationMapTest()
      : data(new double[15]),
        size(15 * sizeof(double)),
        record({data, size, nullptr})
  {
  }

  virtual ~AllocationMapTest()
  {
    delete[] data;
  }

  void TearDown() override
  {
    map.clear();
  }

  umpire::util::AllocationMap map;

  double* data;
  std::size_t size;
  umpire::util::AllocationRecord record;
};

TEST_F(AllocationMapTest, Add)
{
  EXPECT_NO_THROW(map.insert(data, record));
}

TEST_F(AllocationMapTest, FindNotFound)
{
  ASSERT_THROW(map.find(data), umpire::util::Exception);
}

TEST_F(AllocationMapTest, Find)
{
  EXPECT_NO_THROW(map.insert(data, record));

  auto actual_record = map.find(data);

  ASSERT_EQ(record, *actual_record);
}

TEST_F(AllocationMapTest, FindOffset)
{
  EXPECT_NO_THROW(map.insert(data, record));

  auto actual_record = map.find(&data[4]);

  ASSERT_EQ(record, *actual_record);
}

TEST_F(AllocationMapTest, Contains)
{
  EXPECT_NO_THROW(map.insert(data, record));

  ASSERT_TRUE(map.contains(data));
}

TEST_F(AllocationMapTest, NotContains)
{
  ASSERT_FALSE(map.contains(data));
}

TEST_F(AllocationMapTest, Remove)
{
  EXPECT_NO_THROW({
    map.insert(data, record);
    map.remove(data);
  });

  ASSERT_THROW(map.find(data), umpire::util::Exception);

  ASSERT_FALSE(map.contains(data));
}

TEST_F(AllocationMapTest, RemoveNotFound)
{
  ASSERT_THROW(map.remove(data), umpire::util::Exception);
}

TEST_F(AllocationMapTest, RemoveAndUse)
{
  EXPECT_NO_THROW(map.insert(data, record));

  auto found_record = map.remove(data);

  ASSERT_EQ(record, found_record);
}

TEST_F(AllocationMapTest, RegisterMultiple)
{
  umpire::util::AllocationRecord next_record{data, 1, nullptr};

  ASSERT_NO_THROW(map.insert(data, record); map.insert(data, next_record););
}

TEST_F(AllocationMapTest, RegisterNone)
{
  auto begin = map.begin(), end = map.end();

  ASSERT_EQ(begin, end);
  ASSERT_EQ(map.size(), 0);
}

TEST_F(AllocationMapTest, RegisterMultipleIteratorSize)
{
  umpire::util::AllocationRecord next_record{data, 1, nullptr};
  umpire::util::AllocationRecord another_record{data + 10, 5, nullptr};

  ASSERT_NO_THROW(map.insert(data, record); map.insert(data, next_record);
                  map.insert(data, another_record););

  std::size_t sz = 0;
  auto iter = map.begin(), end = map.end();
  while (iter != end) {
    ++sz;
    ++iter;
  }
  ASSERT_EQ(sz, 3);
}

TEST_F(AllocationMapTest, RegisterNoneIteratorSize)
{
  std::size_t sz = 0;
  auto iter = map.begin(), end = map.end();
  while (iter != end) {
    ++sz;
    ++iter;
  }
  ASSERT_EQ(sz, 0);
}

TEST_F(AllocationMapTest, FindMultiple)
{
  umpire::util::AllocationRecord next_record{data, 1, nullptr};

  EXPECT_NO_THROW({
    map.insert(data, record);
    map.insert(data, next_record);
  });

  auto actual_record = map.find(data);
  ASSERT_EQ(next_record, *actual_record);

  map.remove(data);

  EXPECT_NO_THROW(actual_record = map.find(data););

  ASSERT_EQ(*actual_record, record);
}

TEST_F(AllocationMapTest, Print)
{
  umpire::util::AllocationRecord next_record{data, 1, nullptr};

  auto extra_data = new double[10];
  umpire::util::AllocationRecord extra_record{extra_data, 10, nullptr};

  map.insert(data, record);
  map.insert(data, next_record);
  map.insert(extra_data, extra_record);

  map.printAll();

  map.print([this](const umpire::util::AllocationRecord& r) {
    return r.ptr == data;
  });

  delete[] extra_data;
}
