//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SyclMemPrefetchOperation_HPP
#define UMPIRE_SyclMemPrefetchOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

class SyclMemPrefetchOperation : public MemoryOperation {
 public:
  void apply(void* src_ptr, umpire::util::AllocationRecord* src_allocation, int value, std::size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_SyclMemPrefetchOperation_HPP
