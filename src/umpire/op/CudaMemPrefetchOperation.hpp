//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaMemPrefetchOperation_HPP
#define UMPIRE_CudaMemPrefetchOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

class CudaMemPrefetchOperation : public MemoryOperation {
 public:
  void apply(void* src_ptr, umpire::util::AllocationRecord* src_allocation, int value, std::size_t length);

  camp::resources::EventProxy<camp::resources::Resource> apply_async(void* src_ptr, util::AllocationRecord* ptr,
                                                                     int value, std::size_t length,
                                                                     camp::resources::Resource& ctx);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_CudaMemPrefetchOperation_HPP
