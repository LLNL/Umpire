//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaAdviseOperation_HPP
#define UMPIRE_CudaAdviseOperation_HPP

#include <cuda_runtime.h>

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

class CudaAdviseOperation : public MemoryOperation {
 public:
  CudaAdviseOperation(cudaMemoryAdvise m_advice);

  /*!
   * @copybrief MemoryOperation::apply
   *
   * Uses cudaMemAdvise to set data as accessed by the appropriate device.
   *
   * @copydetails MemoryOperation::apply
   */
  void apply(void* src_ptr, util::AllocationRecord* src_allocation, int val, std::size_t length);

 private:
  cudaMemoryAdvise m_advice;
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_CudaAdviseOperation_HPP
