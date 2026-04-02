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
class memory;

/*!
 * \brief Metadata captured for a tracked API v2 allocation.
 *
 * Allocation records are stored by `detail::registry` so tools and
 * interoperability layers can recover the size and owning memory object for a
 * live pointer.
 */
struct allocation_record {
  /*!
   * \brief Construct a record for a live allocation.
   *
   * \param p Allocation base pointer.
   * \param s Requested size in bytes.
   * \param strat Memory resource or strategy responsible for the allocation.
   */
  allocation_record(void* p, std::size_t s, memory* strat) : ptr{p}, size{s}, strategy{strat}
  {
  }

  //! Construct an empty placeholder record.
  allocation_record() : ptr{nullptr}, size{0}, strategy{nullptr}
  {
  }

  //! Base pointer returned by the allocation.
  void* ptr;
  //! Requested allocation size in bytes.
  std::size_t size;
  //! Owning memory object that created the allocation.
  memory* strategy;

#if defined(UMPIRE_ENABLE_BACKTRACE)
  //! Allocation-site backtrace when backtrace collection is enabled.
  util::backtrace allocation_backtrace;
#endif // UMPIRE_ENABLE_BACKTRACE
};

} // namespace umpire

#endif // UMPIRE_allocation_record_HPP
