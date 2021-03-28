//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "umpire/tpl/judy/judy.h"

struct JudyTest {
  void addkey(uint64_t key)
  {
    // judy_cell:  insert a string into the judy array, return cell pointer.
    JudySlot* cell{ judy_cell(array, reinterpret_cast<unsigned char*>(&key), key_length) };

    if ( cell == nullptr ) {
      std::cout << std::setfill('0') << std::setw(16) << std::hex << key;
    }

    assert( cell != nullptr && "addkey: judy_cell returned NULL!" );
    uint64_t* val{ reinterpret_cast<uint64_t*>(cell) };
    *val = key+1;    // Just make up a non-zero value, doesn't matter what
  }

  void check_value_and_key(uint64_t key)
  {
    // judy_strt is called by umpire (doFindOrBefore).  For testing purposes, I
    // am checking that I can get the value from both judy_strt and judy_slot.
    // I have confirmed that I am able to obtain the value from both judy_strt
    // and judy_slot below.  However, neither call seems to be setting up the
    // judy-stack adequately to allow the key to be successfully obtained.
    // Strange... This only happens when enough (consecutive) keys have been
    // generated to cause all of the array nodes to be decomposed into a full
    // radix tree (and only on rzansel!).  This is the reason for UM-851.
    //
    {
      uint64_t k{ key };

      // judy_strt: retrieve the cell pointer greater than or equal to given key
      JudySlot* cell{
        judy_strt(array, reinterpret_cast<unsigned char*>(&k), key_length) };
      assert(cell != nullptr && "cell from judy_strt NULL!");

      // judy_key:   retrieve the string value for the most recent judy query.
      uint64_t* val{ reinterpret_cast<uint64_t*>(cell) };
      assert(*val == key+1 && "value from judy_strt is wrong!"); // WILL FAIL

      // k = 0;
      judy_key(array, reinterpret_cast<unsigned char*>(&k), key_length);
      assert(k == key && "Key from judy_strt wrong!");
    }

    {
      uint64_t k{ key };

      // judy_slot: retrieve the cell pointer, or return NULL for a given key.
      JudySlot* cell{
        judy_slot(array, reinterpret_cast<unsigned char*>(&k), key_length) };
      assert(cell != nullptr && "cell from judy_slot NULL!");

      // judy_key:   retrieve the string value for the most recent judy query.
      uint64_t* val{ reinterpret_cast<uint64_t*>(cell) };
      assert(*val == key+1 && "value from judy_slot is wrong!"); // WILL FAIL

      // k = 0;
      judy_key(array, reinterpret_cast<unsigned char*>(&k), key_length);
      assert(k == key && "Key from judy_strt wrong!");
    }
  }

  //  judy_open:  open a new judy array returning a judy object.
  JudyTest() : array{ judy_open(key_length, key_depth) }
  {
  }

  ~JudyTest()
  {
    judy_close(array);
  }

  const unsigned int key_depth{1};
  const unsigned int key_length{sizeof(uintptr_t)};
  const uint64_t bad_judy_base{ 0x0000200000070000 };
  Judy* array;
};

int main(int, char**)
{
  JudyTest judy;
  uint64_t key;

  for (key = judy.bad_judy_base; key < judy.bad_judy_base+0x10000; key++) {
    judy.addkey(key);
  }

  for (key = judy.bad_judy_base; key < judy.bad_judy_base+0x10000; key++) {
    judy.check_value_and_key(key);
  }

  return 0;
}
