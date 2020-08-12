//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_OpenMPTargetMemsetOperation_HPP
#define UMPIRE_OpenMPTargetMemsetOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

class OpenMPTargetMemsetOperation : public MemoryOperation {
 public:
  /*
   * \copybrief MemoryOperation::apply
   *
   * \copydetails MemoryOperation::apply
   */
  void apply(void* src_ptr, umpire::util::AllocationRecord* src_allocation,
             int value, std::size_t length);

 private:
  int m_device_id;
};

} // namespace op
} // end of namespace umpire

#endif // UMPIRE_OpenMPTargetMemsetOperation_HPP
