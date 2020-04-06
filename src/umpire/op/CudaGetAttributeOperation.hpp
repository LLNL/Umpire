//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaGetAttributeOperation_HPP
#define UMPIRE_CudaGetAttributeOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

#include <cuda_runtime_api.h>

namespace umpire {
namespace op {

/*!
 * \brief Copy operation to move data from CPU to NVIDIA GPU memory.
 */
template<cudaMemRangeAttribute ATTRIBUTE>
class CudaGetAttributeOperation : public MemoryOperation {
 public:
   /*!
    * @copybrief MemoryOperation::check_apply
    *
    * Uses cudaMemRangeGetAtribute to check attributes of a CUDA memory range.
    *
    * @copydetails MemoryOperation::transform
    */
  bool check_apply(
      void* src_ptr,
      umpire::util::AllocationRecord *src_allocation,
      int val,
      std::size_t length) override;
};

#include "umpire/op/CudaGetAttributeOperation.inl"

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_CudaGetAttributeOperation_HPP
