//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SyclCopyFromOperation_HPP
#define UMPIRE_SyclCopyFromOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Copy operation to move data between a Intel GPU and CPU memory.
 */
class SyclCopyFromOperation : public MemoryOperation {
 public:
  /*!
   * @copybrief MemoryOperation::transform
   *
   * Uses SYCL memcpy to move data when src_ptr is on Intel GPU and dst_ptr
   * is on CPU
   *
   * @copydetails MemoryOperation::transform
   */
  void transform(void* src_ptr, void** dst_ptr, util::AllocationRecord* src_allocation,
                 util::AllocationRecord* dst_allocation, std::size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_SyclCopyFromOperation_HPP
