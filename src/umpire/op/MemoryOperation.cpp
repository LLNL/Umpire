//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/MemoryOperation.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

void MemoryOperation::transform(void* UMPIRE_UNUSED_ARG(src_ptr), void** UMPIRE_UNUSED_ARG(dst_ptr),
                                util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
                                util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
                                std::size_t UMPIRE_UNUSED_ARG(length))
{
  UMPIRE_ERROR(runtime_error, "MemoryOperation::transform() is not implemented");
}

camp::resources::EventProxy<camp::resources::Resource> MemoryOperation::transform_async(
    void* src_ptr, void** dst_ptr, util::AllocationRecord* src_allocation, util::AllocationRecord* dst_allocation,
    std::size_t length, camp::resources::Resource& ctx)
{
  UMPIRE_LOG(Warning, "MemoryOperation::transform_async() calling synchronous transform()");
  ctx.get_event().wait();
  this->transform(src_ptr, dst_ptr, src_allocation, dst_allocation, length);
  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

void MemoryOperation::apply(void* UMPIRE_UNUSED_ARG(src_ptr), util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
                            int UMPIRE_UNUSED_ARG(val), std::size_t UMPIRE_UNUSED_ARG(length))
{
  UMPIRE_ERROR(runtime_error, "MemoryOperation::apply() is not implemented");
}

camp::resources::EventProxy<camp::resources::Resource> MemoryOperation::apply_async(
    void* src_ptr, util::AllocationRecord* src_allocation, int val, std::size_t length, camp::resources::Resource& ctx)
{
  UMPIRE_LOG(Warning, "MemoryOperation::apply_async() calling synchronous apply()");
  ctx.get_event().wait();
  this->apply(src_ptr, src_allocation, val, length);
  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
