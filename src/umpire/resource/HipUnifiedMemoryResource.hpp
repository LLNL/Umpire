//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipUnifiedMemoryResource_HPP
#define UMPIRE_HipUnifiedMemoryResource_HPP

#include "umpire/alloc/HipMallocManagedAllocator.hpp"
#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Concrete MemoryResource object that uses the template _allocator to
 * allocate and deallocate memory.
 */
class HipUnifiedMemoryResource : public MemoryResource {
 public:
  HipUnifiedMemoryResource(Platform platform, const std::string& name, int id, MemoryResourceTraits traits);

  void* allocate(std::size_t bytes);
  void deallocate(void* ptr, std::size_t size);

  bool isAccessibleFrom(Platform p) noexcept;
  Platform getPlatform() noexcept;

 protected:
  alloc::HipMallocManagedAllocator m_allocator;

  Platform m_platform;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_HipUnifiedMemoryResource_HPP
