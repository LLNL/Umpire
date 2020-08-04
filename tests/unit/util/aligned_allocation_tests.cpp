//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/util/AlignedAllocation.hpp"

#include "gtest/gtest.h"

#include <iostream>

TEST(AligndAllocation, Alignment)
{
#ifdef need_a_good_alignment_tester
  // This also belongs in the pool tests
  const std::size_t bufsize{512};
  const std::size_t alignment{128};
  umpire::util::AlignedAllocation align{alignment};

  for (std::size_t i = 0; i <= bufsize; i += alignment) {
    int start{ i == 0 ? 0 : -2 };
    for (int j = start; j <= 2; ++j) {
      std::size_t size{bufsize};
      void* buffer{(void*)(i+j)};
      align.align_create(size, buffer);

      std::size_t original_size;
      void* original_buffer;

      align.align_destroy(buffer, original_size, original_buffer);

      std::size_t align_check{align.round_up(i+j)};

      EXPECT_EQ(buffer, (void*)align_check);

      std::cout
        << "alignment: " << alignment << " "
        << "base: " << original_buffer << " "
        << "aligned pointer: " << buffer << " "
        << "size: " << bufsize << " "
        << "adjusted size: " << size << " "
        << "round_up(" << (i+j) << "): " << align.round_up(i+j) << " "
        << std::endl;
    }
  }
#endif // need_a_good_alignment_tester
}
