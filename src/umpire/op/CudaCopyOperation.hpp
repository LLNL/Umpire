//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaCopyOperation_HPP
#define UMPIRE_CudaCopyOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Copy operation to move data between two GPU addresses.
 */
class CudaCopyOperation : public MemoryOperation {
 public:
  /*!
   * @copybrief MemoryOperation::transform
   *
   * Uses cudaMemcpy to move data when both src_ptr  and dst_ptr are on NVIDIA
   * GPUs.
   *
   * @copydetails MemoryOperation::transform
   */
  void transform(void* src_ptr, void** dst_ptr,
                 umpire::util::AllocationRecord* src_allocation,
                 umpire::util::AllocationRecord* dst_allocation,
                 std::size_t length);

  camp::resources::Event transform_async(void* src_ptr, void** dst_ptr,
                                         util::AllocationRecord* src_allocation,
                                         util::AllocationRecord* dst_allocation,
                                         std::size_t length,
                                         camp::resources::Resource& ctx);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_CudaCopyOperation_HPP
