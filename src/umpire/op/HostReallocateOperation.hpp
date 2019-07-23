//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HostReallocateOperation_HPP
#define UMPIRE_HostReallocateOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Reallocate data in CPU memory.
 */
class HostReallocateOperation :
  public MemoryOperation {
 public:
  /*!
   * \copybrief MemoryOperation::transform
   *
   * Uses POSIX realloc to reallocate memory in the CPU memory.
   *
   * \copydetails MemoryOperation::transform
   */
  void transform(
      void* src_ptr,
      void** dst_ptr,
      util::AllocationRecord *src_allocation,
      util::AllocationRecord *dst_allocation,
      std::size_t length);
};

} // end of naemspace op
} // end of namespace umpire

#endif // UMPIRE_HostReallocateOperation_HPP
