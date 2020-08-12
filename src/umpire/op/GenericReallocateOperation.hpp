//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_GenericReallocateOperation_HPP
#define UMPIRE_GenericReallocateOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * Generic reallocate operation to work on any current_ptr location.
 */
class GenericReallocateOperation : public MemoryOperation {
 public:
  /*!
   * \copybrief MemoryOperation::transform
   *
   * This operation relies on ResourceManager::copy,
   * AllocationStrategy::allocate and AllocationStrategy::deallocate to
   * implement a reallocate operation that can work for any current_ptr
   * location.
   *
   * \copydetails MemoryOperation::transform
   */
  void transform(void* current_ptr, void** new_ptr,
                 util::AllocationRecord* current_allocation,
                 util::AllocationRecord* new_allocation, std::size_t new_size);
};

} // namespace op
} // end of namespace umpire

#endif // UMPIRE_GenericReallocateOperation_HPP
