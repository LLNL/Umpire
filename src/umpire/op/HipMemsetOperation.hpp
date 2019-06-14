//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#ifndef UMPIRE_HipMemsetOperation_HPP
#define UMPIRE_HipMemsetOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Memset on AMD device memory.
 */
class HipMemsetOperation : public MemoryOperation {
 public:
   /*!
    * @copybrief MemoryOperation::apply
    *
    * Uses hipMemset to set first length bytes of src_ptr to value.
    *
    * @copydetails MemoryOperation::apply
    */
  void apply(
      void* src_ptr,
      util::AllocationRecord* ptr,
      int value,
      size_t length);
};

} // end of naemspace op
} // end of namespace umpire

#endif // UMPIRE_HipMemsetOperation_HPP
