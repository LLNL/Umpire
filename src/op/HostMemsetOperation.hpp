//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
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
  void apply(
      void* src_ptr,
      util::AllocationRecord* allocation,
      int value,
      size_t length);
};

} // end of naemspace op
} //end of namespace umpire

#endif // UMPIRE_HostMemsetOperation_HPP
