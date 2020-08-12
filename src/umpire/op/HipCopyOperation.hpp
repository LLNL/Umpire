//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipCopyOperation_HPP
#define UMPIRE_HipCopyOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Copy operation to move data between two GPU addresses.
 */
class HipCopyOperation : public MemoryOperation {
 public:
  /*!
   * @copybrief MemoryOperation::transform
   *
   * Uses hipMemcpy to move data when both src_ptr  and dst_ptr are on AMD
   * GPUs.
   *
   * @copydetails MemoryOperation::transform
   */
  void transform(void* src_ptr, void** dst_ptr,
                 umpire::util::AllocationRecord* src_allocation,
                 umpire::util::AllocationRecord* dst_allocation,
                 std::size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_HipCopyOperation_HPP
