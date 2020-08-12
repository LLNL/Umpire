//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
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
class HostReallocateOperation : public MemoryOperation {
 public:
  /*!
   * \copybrief MemoryOperation::transform
   *
   * Uses POSIX realloc to reallocate memory in the CPU memory.
   *
   * \copydetails MemoryOperation::transform
   */
  void transform(void* current_ptr, void** new_ptr,
                 util::AllocationRecord* current_allocation,
                 util::AllocationRecord* new_allocation, std::size_t new_size);
};

} // namespace op
} // end of namespace umpire

#endif // UMPIRE_HostReallocateOperation_HPP
