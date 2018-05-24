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
      size_t length);
};

} // end of naemspace op
} // end of namespace umpire

#endif // UMPIRE_HostReallocateOperation_HPP

