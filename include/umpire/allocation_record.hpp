//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_allocation_record_HPP
#define UMPIRE_allocation_record_HPP

#include <cstddef>

#include "umpire/util/backtrace.hpp"

namespace umpire {

namespace strategy {
class AllocationStrategy;
}

struct allocation_record {
  allocation_record(void* p, std::size_t s, strategy::AllocationStrategy* strat) : ptr{p}, size{s}, strategy{strat}
  {
  }

  allocation_record() : ptr{nullptr}, size{0}, strategy{nullptr}
  {
  }

  void* ptr;
  std::size_t size;
  strategy::AllocationStrategy* strategy;

#if defined(UMPIRE_ENABLE_BACKTRACE)
  util::backtrace allocation_backtrace;
#endif // UMPIRE_ENABLE_BACKTRACE
};

} // namespace umpire

#endif // UMPIRE_allocation_record_HPP
