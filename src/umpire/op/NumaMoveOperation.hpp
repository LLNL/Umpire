//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_NumaMoveOperation_HPP
#define UMPIRE_NumaMoveOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Relocate a pointer to a different NUMA node.
 */
class NumaMoveOperation : public MemoryOperation {
 public:
  /*
   * \copybrief MemoryOperation::transform
   *
   * Relocate the memory to the NUMA node implicitly given by the
   * dst_allocation. This performs a static_cast to a a NumaPolicy,
   * and will therefore throw an error if dst_allocation is not this
   * type. dst_ptr == src_ptr when successful.
   *
   * \copydetails MemoryOperation::transform
   */
  void transform(void* src_ptr, void** dst_ptr,
                 umpire::util::AllocationRecord* src_allocation,
                 umpire::util::AllocationRecord* dst_allocation,
                 std::size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_NumaMoveOperation_HPP
