//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipPinnedMemoryResource_HPP
#define UMPIRE_HipPinnedMemoryResource_HPP

#include "umpire/alloc/HipPinnedAllocator.hpp"
#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Concrete MemoryResource object that uses the template _allocator to
 * allocate and deallocate memory.
 */
class HipPinnedMemoryResource : public MemoryResource {
 public:
  HipPinnedMemoryResource(Platform platform, const std::string& name, int id, MemoryResourceTraits traits);

  void* allocate(std::size_t bytes);
  void deallocate(void* ptr, std::size_t size);

  bool isAccessibleFrom(Platform p) noexcept;
  Platform getPlatform() noexcept;

 protected:
  alloc::HipPinnedAllocator m_allocator;

  Platform m_platform;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_HipPinnedMemoryResource_HPP
