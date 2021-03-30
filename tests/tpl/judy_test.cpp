//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

#include <cstdint>
#include <sstream>
#include <string>

#include "umpire/tpl/judy/judy.h"

class JudyTest : public ::testing::Test {
 protected:
  //  judy_open:  open a new judy array returning a judy object.
  JudyTest() : array{ judy_open(key_length, key_depth) }
  {
  }

  ~JudyTest()
  {
    judy_close(array);
  }

  void TearDown() override
  {
  }

  const unsigned int key_depth{1};
  const unsigned int key_length{sizeof(uintptr_t)};
  const uint64_t bad_judy_base{ 0x0000200000070000 };
  const uint64_t max_keys{1000};
  Judy* array;
};

TEST_F(JudyTest, Construction)
{
  EXPECT_TRUE(array != nullptr);
}

TEST_F(JudyTest, WriteValues)
{
  EXPECT_TRUE(array != nullptr);

  for (uint64_t key = bad_judy_base; key < bad_judy_base+max_keys; key++) {
    uint64_t k{ key };

    // judy_cell:  insert a string into the judy array, return cell pointer.
    JudySlot* cell{ judy_cell(array, reinterpret_cast<unsigned char*>(&k), key_length) };
    EXPECT_TRUE(cell != nullptr);

    uint64_t* val{ reinterpret_cast<uint64_t*>(cell) };
    *val = key+1;    // Just make up a non-zero value, doesn't matter what
  }

  for (uint64_t key = bad_judy_base; key < bad_judy_base+max_keys; key++) {
    {
      uint64_t k{ key };

      // judy_strt: retrieve the cell pointer greater than or equal to given key
      JudySlot* cell{ judy_strt(array, reinterpret_cast<unsigned char*>(&k), key_length) };
      EXPECT_TRUE(cell != nullptr);

      uint64_t* val{ reinterpret_cast<uint64_t*>(cell) };

      EXPECT_EQ(*val, key+1);

      k = 0;
      judy_key(array, reinterpret_cast<unsigned char*>(&k), key_length);

      EXPECT_EQ(k, key );
    }

    {
      uint64_t k{ key };

      // judy_slot: retrieve the cell pointer, or return NULL for a given key.
      JudySlot* cell{
        judy_slot(array, reinterpret_cast<unsigned char*>(&k), key_length) };
      EXPECT_TRUE(cell != nullptr);

      // judy_key:   retrieve the string value for the most recent judy query.
      uint64_t* val{ reinterpret_cast<uint64_t*>(cell) };

      EXPECT_EQ(*val, key+1);

      k = 0;
      judy_key(array, reinterpret_cast<unsigned char*>(&k), key_length);
      EXPECT_EQ(k, key );
    }
  }
}
