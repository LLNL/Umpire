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
#ifndef UMPIRE_CudaCopyToOperation_HPP
#define UMPIRE_CudaCopyToOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Copy operation to move data from CPU to NVIDIA GPU memory.
 */
class CudaCopyToOperation : public MemoryOperation {
 public:
   /*!
    * @copybrief MemoryOperation::transform
    *
    * Uses cudaMemcpy to move data when src_ptr is on the CPU and dst_ptr
    * is on an NVIDIA GPU.
    *
    * @copydetails MemoryOperation::transform
    */
  void transform(
      void* src_ptr,
      void* dst_ptr,
      umpire::util::AllocationRecord *src_allocation,
      umpire::util::AllocationRecord *dst_allocation,
      size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_CudaCopyToOperation_HPP
