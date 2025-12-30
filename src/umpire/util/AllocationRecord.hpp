//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_AllocationRecord_HPP
#define UMPIRE_AllocationRecord_HPP

#include <cstddef>
#include <memory>
#include <string>

#include "umpire/util/backtrace.hpp"

namespace umpire {

namespace strategy {
class AllocationStrategy;
}

namespace util {

struct AllocationRecord {
  AllocationRecord(void* p, std::size_t s, strategy::AllocationStrategy* strat)
      : ptr{p}, size{s}, strategy{strat}, name{}
  {
  }

  AllocationRecord(void* p, std::size_t s, strategy::AllocationStrategy* strat, const std::string& _name)
      : ptr{p}, size{s}, strategy{strat}, name{_name}
  {
  }

  AllocationRecord() : ptr{nullptr}, size{0}, strategy{nullptr}, name{}
  {
  }

  void* ptr;
  std::size_t size;
  strategy::AllocationStrategy* strategy;
  std::string name;
#if defined(UMPIRE_ENABLE_BACKTRACE)
  util::backtrace allocation_backtrace;
#endif // UMPIRE_ENABLE_BACKTRACE
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocationRecord_HPP
