//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SyclMemsetOperation_HPP
#define UMPIRE_SyclMemsetOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Memset on Intel GPU device memory.
 */
class SyclMemsetOperation : public MemoryOperation {
 public:
  /*!
   * @copybrief MemoryOperation::apply
   *
   * Uses SYCL memset to set first length bytes of src_ptr to value.
   *
   * @copydetails MemoryOperation::apply
   */
  void apply(void* src_ptr, util::AllocationRecord* ptr, int value,
             std::size_t length);
};

} // namespace op
} // end of namespace umpire

#endif // UMPIRE_SyclMemsetOperation_HPP
