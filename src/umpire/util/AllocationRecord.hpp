//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_AllocationRecord_HPP
#define UMPIRE_AllocationRecord_HPP

#include <cstddef>

#include <memory>

#include "umpire/util/Backtrace.hpp"

#ifdef UMPIRE_ENABLE_ALLOCATION_BACKTRACE
#define UMPIRE_GET_ALLOCATION_BACKTRACE(record)  record.allocationBacktrace.getBacktrace()
#else
#define UMPIRE_GET_ALLOCATION_BACKTRACE(record)
#endif

namespace umpire {

namespace strategy {
  class AllocationStrategy;
}

namespace util {

struct AllocationRecord
{
  AllocationRecord(void* p, std::size_t s, strategy::AllocationStrategy* strat)
    : ptr{p}, size{s}, strategy{strat} { };

  AllocationRecord()
    : ptr{nullptr}, size{0}, strategy{nullptr} { };

  void* ptr;
  std::size_t size;
  strategy::AllocationStrategy* strategy;
#ifdef UMPIRE_ENABLE_ALLOCATION_BACKTRACE
  util::Backtrace allocationBacktrace;
#endif // UMPIRE_ENABLE_ALLOCATION_BACKTRACE
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocationRecord_HPP
