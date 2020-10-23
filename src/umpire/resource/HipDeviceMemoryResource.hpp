//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipDeviceMemoryResource_HPP
#define UMPIRE_HipDeviceMemoryResource_HPP

#include "umpire/alloc/HipMallocAllocator.hpp"
#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Concrete MemoryResource object that uses the template _allocator to
 * allocate and deallocate memory.
 */
class HipDeviceMemoryResource : public MemoryResource {
 public:
  HipDeviceMemoryResource(Platform platform, const std::string& name, int id,
                           MemoryResourceTraits traits);

  void* allocate(std::size_t bytes);
  void deallocate(void* ptr);

  std::size_t getCurrentSize() const noexcept;
  std::size_t getHighWatermark() const noexcept;

  bool isAccessibleFrom(Platform p) noexcept;
  Platform getPlatform() noexcept;

 protected:
  alloc::HipMallocAllocator m_allocator;

  Platform m_platform;

 private:
  bool isHostPageable(); 
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_HipDeviceMemoryResource_HPP
