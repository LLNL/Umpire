//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HostMemsetOperation_HPP
#define UMPIRE_HostMemsetOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Memset an allocation in CPU memory.
 */
class HostMemsetOperation : public MemoryOperation {
 public:
  /*!
   * \copybrief MemoryOperation::apply
   *
   * Uses std::memset to set the first length bytes of src_ptr to value.
   *
   * \copydetails MemoryOperation::apply
   */
  void apply(void* src_ptr, util::AllocationRecord* allocation, int value,
             std::size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_HostMemsetOperation_HPP
