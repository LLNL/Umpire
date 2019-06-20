//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipConstantMemoryResource_HPP
#define UMPIRE_HipConstantMemoryResource_HPP

#include "umpire/resource/MemoryResource.hpp"

#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Platform.hpp"

#include <hip/hip_runtime.h>

__constant__ char umpire_internal_device_constant_memory[64*1024];

namespace umpire {
namespace resource {


class HipConstantMemoryResource :
  public MemoryResource
{
  public:
    HipConstantMemoryResource(const std::string& name, int id, MemoryResourceTraits traits);

    void* allocate(std::size_t bytes);
    void deallocate(void* ptr);

    long getCurrentSize() const noexcept;
    long getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;

  private:
    long m_current_size;
    long m_highwatermark;

    Platform m_platform;

    std::size_t m_offset;
    void* m_ptr;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_HipConstantMemoryResource_HPP
