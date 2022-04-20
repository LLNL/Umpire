//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipAdviseOperation_HPP
#define UMPIRE_HipAdviseOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

#include <hip/hip_runtime.h>

namespace umpire {
namespace op {

class HipAdviseOperation : public MemoryOperation {
 public:
 HipAdviseOperation(hipMemoryAdvise a);
  /*!
   * @copybrief MemoryOperation::apply
   *
   * Uses cudaMemAdvise to set data as accessed by the appropriate device.
   *
   * @copydetails MemoryOperation::apply
   */
  void apply(void* src_ptr, util::AllocationRecord* src_allocation, int val, std::size_t length);
  
  private:
  hipMemoryAdvise m_advise;
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_HipAdviseOperation_HPP
