//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <cassert>
#include <cstdint>

#include "umpire/tpl/judy/judy.h"

namespace {
  static constexpr unsigned int judy_depth{1}; // Judy: number of Integers in a key
  static constexpr unsigned int judy_max_levels{sizeof(uintptr_t)}; // Judy: max height of stack
  static constexpr unsigned int judy_max{judy_depth * JUDY_key_size}; // Judy: length of key in bytes
  static constexpr uint64_t bad_judy_base{ 0x0000200000070000 };
}

void addkey(Judy* array, uint64_t key)
{
  JudySlot* cell{ judy_cell(array, reinterpret_cast<unsigned char*>(&key), judy_max) };
  assert( cell != nullptr && "addkey: judy_cell returned NULL!" );
  uint64_t* val{ reinterpret_cast<uint64_t*>(cell) };
  *val = key+1;    // Just make sure Val is non-zero
}

uint64_t getval(Judy* array, uint64_t key)
{
  // JudySlot* cell{ judy_strt(array, reinterpret_cast<unsigned char*>(&key), judy_max) };
  JudySlot* cell{ judy_slot(array, reinterpret_cast<unsigned char*>(&key), judy_max) };
  assert( cell != nullptr && "getval: judy_cell returned NULL!" );

  uint64_t* val{ reinterpret_cast<uint64_t*>(cell) };
  assert(*val == key+1 && "getval: value is incorect!");

  // confirm the key above was indeed found
  uint64_t key_check{0};
  judy_key(array, reinterpret_cast<unsigned char*>(&key_check), judy_max);
  assert( key == key_check && "Key not found!");

  return *val;
}

int main(int, char**)
{
  Judy* array{ judy_open(judy_max_levels, judy_depth) };
  uint64_t key;

  for (key = bad_judy_base; key < bad_judy_base+0x10000; key++) {
    addkey(array, key);
  }

  for (key = bad_judy_base; key < bad_judy_base+0x10000; key++) {
    getval(array, key);
  }

  judy_close(array);
  return 0;
}
