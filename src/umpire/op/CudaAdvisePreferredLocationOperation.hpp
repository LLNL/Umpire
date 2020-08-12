//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaAdvisePreferredLocationOperation_HPP
#define UMPIRE_CudaAdvisePreferredLocationOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

class CudaAdvisePreferredLocationOperation : public MemoryOperation {
 public:
  /*!
   * @copybrief MemoryOperation::apply
   *
   * Uses cudaMemAdvise to set preffered location of data.
   *
   * @copydetails MemoryOperation::apply
   */
  void apply(void* src_ptr, util::AllocationRecord* src_allocation, int val,
             std::size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_CudaAdvisePreferredLocationOperation_HPP
