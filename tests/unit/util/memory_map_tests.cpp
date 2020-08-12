//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/util/MemoryMap.hpp"

using Map = umpire::util::MemoryMap<int>;

static std::size_t size_by_iterator(Map& map)
{
  auto iter = map.begin(), end = map.end();
  std::size_t size = 0;
  while (iter != end) {
    ++size;
    ++iter;
  }
  return size;
}

class MemoryMapTest : public ::testing::Test {
 protected:
  MemoryMapTest()
      : a{reinterpret_cast<void*>(1)},
        b{reinterpret_cast<void*>(2)},
        c{reinterpret_cast<void*>(3)}
  {
  }

  void TearDown() override
  {
    map.clear();
  }

  Map map;
  void* a;
  void* b;
  void* c;
};

TEST_F(MemoryMapTest, insert)
{
  Map::Iterator iter{map.end()};
  bool inserted;

  {
    std::tie(iter, inserted) = map.insert(a, 1);
    EXPECT_EQ(iter, map.begin());
    EXPECT_EQ(inserted, true);
  }

  {
    std::tie(iter, inserted) = map.insert(a, 1);
    EXPECT_EQ(iter, map.begin());
    EXPECT_EQ(inserted, false);
  }

  EXPECT_NO_THROW({ map.insert(b, 2); });

  ASSERT_EQ(map.size(), 2);
  ASSERT_EQ(size_by_iterator(map), 2);
}

TEST_F(MemoryMapTest, find)
{
  EXPECT_NO_THROW({
    map.insert(a, 1);
    map.insert(b, 2);
  });

  {
    auto iter = map.find(a);
    EXPECT_EQ(iter->first, a);
  }

  {
    auto iter = map.find(c);
    EXPECT_EQ(iter, map.end());
    EXPECT_EQ(iter->second, nullptr);
  }
}

TEST_F(MemoryMapTest, findOrBefore)
{
  EXPECT_NO_THROW(map.insert(a, 1));

  {
    auto iter = map.findOrBefore(static_cast<char*>(a) + 10);
    EXPECT_EQ(iter->first, a);
  }

  map.erase(a);
  {
    auto iter = map.findOrBefore(static_cast<char*>(a) + 10);
    EXPECT_EQ(iter, map.begin());
    EXPECT_EQ(iter, map.end());
  }

  ASSERT_EQ(map.size(), 0);
}

TEST_F(MemoryMapTest, erase)
{
  map.insert(a, 1);

  ASSERT_NO_THROW(map.erase(a));
  ASSERT_ANY_THROW(map.erase(a));
  ASSERT_EQ(map.size(), 0);
}

TEST_F(MemoryMapTest, removeLast)
{
  map.insert(a, 1);

  map.begin();

  ASSERT_NO_THROW(map.removeLast());
  ASSERT_ANY_THROW(map.erase(a));
  ASSERT_EQ(map.size(), 0);
}

TEST_F(MemoryMapTest, clear)
{
  map.insert(a, 1);
  map.insert(b, 2);
  map.insert(c, 3);

  ASSERT_NO_THROW(map.erase(a));

  map.clear();
  ASSERT_EQ(map.size(), 0);
  ASSERT_EQ(size_by_iterator(map), 0);
}

TEST_F(MemoryMapTest, Iterator)
{
  Map::Iterator ia{map.end()};
  bool inserted;
  std::tie(ia, inserted) = map.insert(a, 1);

  auto begin = map.begin();
  ASSERT_EQ(begin->first, a);
  ASSERT_EQ(ia, map.begin());
  ASSERT_EQ(++ia, map.end());
}

TEST_F(MemoryMapTest, ordering)
{
  Map::Iterator ia{map.end()}, ib{map.end()};
  bool inserted;
  std::tie(ia, inserted) = map.insert(a, 1);
  std::tie(ib, inserted) = map.insert(b, 2);

  ASSERT_EQ(++ia, ib);
}
