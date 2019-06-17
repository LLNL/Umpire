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
#ifndef UMPIRE_CudaMemsetOperation_HPP
#define UMPIRE_CudaMemsetOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Memset on NVIDIA device memory.
 */
class CudaMemsetOperation : public MemoryOperation {
 public:
   /*!
    * @copybrief MemoryOperation::apply
    *
    * Uses cudaMemset to set first length bytes of src_ptr to value.
    *
    * @copydetails MemoryOperation::apply
    */
  void apply(
      void* src_ptr,
      util::AllocationRecord* ptr,
      int value,
      std::size_t length);
};

} // end of naemspace op
} //end of namespace umpire

#endif // UMPIRE_CudaMemsetOperation_HPP
