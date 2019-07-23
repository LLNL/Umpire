//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
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
 * Generic reallocate operation to work on any src_ptr location.
 */
class GenericReallocateOperation :
  public MemoryOperation {
 public:
   /*!
    * \copybrief MemoryOperation::transform
    *
    * This operation relies on ResourceManager::copy,
    * AllocationStrategy::allocate and AllocationStrategy::deallocate to
    * implement a reallocate operation that can work for any src_ptr location.
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

#endif // UMPIRE_GenericReallocateOperation_HPP
