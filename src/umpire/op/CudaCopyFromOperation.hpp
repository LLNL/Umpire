//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaCopyFromOperation_HPP
#define UMPIRE_CudaCopyFromOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Copy operation to move data from a NVIDA GPU to CPU memory.
 */
class CudaCopyFromOperation : public MemoryOperation {
 public:
  /*!
   * @copybrief MemoryOperation::transform
   *
   * Uses cudaMemcpy to move data when src_ptr is on a NVIDIA GPU and dst_ptr
   * is on the CPU.
   *
   * @copydetails MemoryOperation::transform
   */
  void transform(void* src_ptr, void** dst_ptr,
                 util::AllocationRecord* src_allocation,
                 util::AllocationRecord* dst_allocation, std::size_t length);

  camp::resources::Event transform_async(void* src_ptr, void** dst_ptr,
                                         util::AllocationRecord* src_allocation,
                                         util::AllocationRecord* dst_allocation,
                                         std::size_t length,
                                         camp::resources::Resource& ctx);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_CudaCopyFromOperation_HPP
