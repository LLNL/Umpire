//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaDeviceMemoryResource_HPP
#define UMPIRE_CudaDeviceMemoryResource_HPP

#include "umpire/alloc/CudaMallocAllocator.hpp"
#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Concrete MemoryResource object that uses the template _allocator to
 * allocate and deallocate memory.
 */
class CudaDeviceMemoryResource : public MemoryResource {
 public:
  CudaDeviceMemoryResource(Platform platform, const std::string& name, int id,
                           MemoryResourceTraits traits);

  void* allocate(std::size_t bytes);
  void deallocate(void* ptr);

  std::size_t getCurrentSize() const noexcept;
  std::size_t getHighWatermark() const noexcept;

  Platform getPlatform() noexcept;

 protected:
  alloc::CudaMallocAllocator m_allocator;

  Platform m_platform;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_CudaDeviceMemoryResource_HPP
