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
#include "umpire/op/RocmMemsetOperation.hpp"


#include <hc.hpp>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void
RocmMemsetOperation::apply(
    void* src_ptr,
    util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation),
    int value,
    size_t length)
{
  unsigned char * cptr = static_cast<unsigned char *>(ptr);
  uint32_t * wptr = static_cast<uint32_t *> ptr;

  char c = static_cast<char>(value);

  uint32_t fill = 
    static_cast<uint32_t>value
    + (static_cast<uint32_t>(value)<<8)
    + (static_cast<uint32_t>(value)<<16)
    + (static_cast<uint32_t>(value)<<24);

  int n = length/4;
  int r = length - n*4;

  if(n+r) {

    hc::extent<1> e(n + (r ? r : 0));

    hc::parallel_for_each(e,  [=] (hc::index<1> idx) [[hc]] {
      if(idx[0] < n) {
        wptr[idx[0]] = fill;
      }
      if(r) {
        if(idx[0] < r) {
          cptr[n*4+idx[0]] = value;
        }
      }
    }).wait();
  }


  UMPIRE_RECORD_STATISTIC(
      "RocmMemsetOperation",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "value", value,
      "size", length,
      "event", "memset");
}

} // end of namespace op
} // end of namespace umpire
