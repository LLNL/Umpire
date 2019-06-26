//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_AllocationRecord_HPP
#define UMPIRE_AllocationRecord_HPP

#include <cstddef>

#include <memory>

namespace umpire {

namespace strategy {
  class AllocationStrategy;
}

namespace util {

struct AllocationRecord
{
  void* ptr;
  std::size_t size;
  strategy::AllocationStrategy* strategy;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocationRecord_HPP
