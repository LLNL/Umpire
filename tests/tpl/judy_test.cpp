//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/tpl/judy/judy.h"

namespace {
  static constexpr unsigned int judy_depth{1}; // Judy: number of Integers in a key
  static constexpr unsigned int judy_max_levels{sizeof(uintptr_t)}; // Judy: max height of stack
  static constexpr unsigned int judy_max{judy_depth * JUDY_key_size}; // Judy: length of key in bytes
  static constexpr uint64_t bad_judy_base{ 0x0000200000070000 };
}

int main(int, char**)
{
  Judy* array{ judy_open(judy_max_levels, judy_depth) };
  JudySlot* last;
  uint64_t ptr;

  for (ptr = bad_judy_base; ptr < bad_judy_base+0x37; ptr++) {
  }

  judy_close(array);
  return 0;
}

