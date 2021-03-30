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
  void write_key_value(const uint64_t key)
  {
    uint64_t k{ key };

    // judy_cell:  insert a string into the judy array, return cell pointer.
    JudySlot* cell{ judy_cell(array, reinterpret_cast<unsigned char*>(&k), key_length) };
    assert( cell != nullptr && "write_key_value: judy_cell returned NULL!" );

    uint64_t* val{ reinterpret_cast<uint64_t*>(cell) };
    *val = key+1;    // Just make up a non-zero value, doesn't matter what

    // perform check immediately, this should always work..
    //
    k = 0;
    judy_key(array, reinterpret_cast<unsigned char*>(&k), key_length);
    if ( k != key ) {
      std::cout
        << "Unexpected Key:"
        << " Expected: " << hex_format(key) << ", "
        << " Returned: " << hex_format(k) << ", "
        << " Cell Location: " << hex_format(reinterpret_cast<uint64_t>(cell)) << ", "
        << " Value: " << hex_format(key+1) << std::endl;
    }
    else {
      std::cout
        << "Key:" << hex_format(key) << ", "
        << "Cell Location: " << hex_format(reinterpret_cast<uint64_t>(cell)) << ", "
        << "Value: " << hex_format(key+1)
        << std::endl;
    }
  }

  void check_value_and_key(uint64_t key)
  {
    {
      uint64_t k{ key };

      // judy_strt: retrieve the cell pointer greater than or equal to given key
      JudySlot* cell{ judy_strt(array, reinterpret_cast<unsigned char*>(&k), key_length) };
      assert(cell != nullptr && "cell from judy_strt NULL!");

      uint64_t* val{ reinterpret_cast<uint64_t*>(cell) };

      if ( *val != key+1 ) {
        std::cout
          << "Unexpected Value from judy_strt:"
          << " Cell Location: " << hex_format(reinterpret_cast<uint64_t>(cell)) << ", "
          << " Expected Value: " << hex_format(key+1) << ", "
          << " Returned Value: " << hex_format(*cell) << std::endl;
      }

      k = 0;
      judy_key(array, reinterpret_cast<unsigned char*>(&k), key_length);

      if ( k != key ) {
        std::cout
          << "Unexpected Key from judy_key after judy_strt:"
          << " Expected: " << hex_format(key) << ", "
          << " Returned: " << hex_format(k) << std::endl;
      }
    }

    {
      uint64_t k{ key };

      // judy_slot: retrieve the cell pointer, or return NULL for a given key.
      JudySlot* cell{
        judy_slot(array, reinterpret_cast<unsigned char*>(&k), key_length) };
      assert(cell != nullptr && "cell from judy_slot NULL!");

      // judy_key:   retrieve the string value for the most recent judy query.
      uint64_t* val{ reinterpret_cast<uint64_t*>(cell) };

      if ( *val != key+1 ) {
        std::cout
          << "Unexpected Value from judy_slot:"
          << " Cell Location: " << hex_format(reinterpret_cast<uint64_t>(cell)) << ", "
          << " Expected Value: " << hex_format(key+1) << ", "
          << " Returned Value: " << hex_format(*cell) << std::endl;
      }

      k = 0;
      judy_key(array, reinterpret_cast<unsigned char*>(&k), key_length);
      if ( k != key ) {
        std::cout
          << "Unexpected Key from judy_key after judy_slot:"
          << " Expected: " << hex_format(key) << ", "
          << " Returned: " << hex_format(k) << std::endl;
      }
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

  std::string hex_format(uint64_t n)
  {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(16) << std::hex << n;
    return ss.str();
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

  for (key = judy.bad_judy_base; key < judy.bad_judy_base+57; key++) {
    judy.write_key_value(key);
  }

  for (key = judy.bad_judy_base; key < judy.bad_judy_base+57; key++) {
    judy.check_value_and_key(key);
  }

  return 0;
}
