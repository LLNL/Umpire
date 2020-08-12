//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipMemsetOperation_HPP
#define UMPIRE_HipMemsetOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Memset on AMD device memory.
 */
class HipMemsetOperation : public MemoryOperation {
 public:
  /*!
   * @copybrief MemoryOperation::apply
   *
   * Uses hipMemset to set first length bytes of src_ptr to value.
   *
   * @copydetails MemoryOperation::apply
   */
  void apply(void* src_ptr, util::AllocationRecord* ptr, int value,
             std::size_t length);
};

} // namespace op
} // end of namespace umpire

#endif // UMPIRE_HipMemsetOperation_HPP
