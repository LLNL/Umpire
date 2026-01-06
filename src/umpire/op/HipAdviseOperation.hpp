//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipAdviseOperation_HPP
#define UMPIRE_HipAdviseOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

class HipAdviseOperation : public MemoryOperation {
 public:
  HipAdviseOperation(hipMemoryAdvise a);

  /*!
   * @copybrief MemoryOperation::apply
   *
   * Uses hipMemAdvise to apply memory advice to the given allocation.
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
