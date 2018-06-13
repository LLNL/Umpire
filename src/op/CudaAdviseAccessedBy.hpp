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
#ifndef UMPIRE_CudaAdviseAccessedBy_HPP
#define UMPIRE_CudaAdviseAccessedBy_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

class CudaAdviseAccessedBy :
  public MemoryOperation {
public:
  /*!
   * @copybrief MemoryOperation::apply
   *
   * Uses cudaMemAdvise to set data as accessed by the appropriate device.
   *
   * @copydetails MemoryOperation::apply
   */
    void apply(
        void* src_ptr,
        util::AllocationRecord *src_allocation,
        int val,
        size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_CudaAdviseAccessedBy_HPP
