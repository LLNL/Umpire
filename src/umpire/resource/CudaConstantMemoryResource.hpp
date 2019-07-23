//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaConstantMemoryResource_HPP
#define UMPIRE_CudaConstantMemoryResource_HPP

#include "umpire/resource/MemoryResource.hpp"

#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Platform.hpp"

#include <cuda_runtime_api.h>

__constant__ char umpire_internal_device_constant_memory[64*1024];

namespace umpire {
namespace resource {


class CudaConstantMemoryResource :
  public MemoryResource
{
  public:
    CudaConstantMemoryResource(const std::string& name, int id, MemoryResourceTraits traits);

    void* allocate(std::size_t bytes);
    void deallocate(void* ptr);

    std::size_t getCurrentSize() const noexcept;
    std::size_t getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;

  private:
    std::size_t m_current_size;
    std::size_t m_highwatermark;

    Platform m_platform;

    std::size_t m_offset;
    void* m_ptr;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_CudaConstantMemoryResource_HPP
