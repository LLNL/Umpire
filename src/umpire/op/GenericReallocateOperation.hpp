//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
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
      size_t length);
};

} // end of naemspace op
} // end of namespace umpire

#endif // UMPIRE_GenericReallocateOperation_HPP

