//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipConstantMemoryResource_HPP
#define UMPIRE_HipConstantMemoryResource_HPP

#include <mutex>

#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace resource {

class HipConstantMemoryResource : public MemoryResource {
 public:
  HipConstantMemoryResource(const std::string& name, int id, MemoryResourceTraits traits);

  void* allocate(std::size_t bytes);
  void deallocate(void* ptr, std::size_t size);

  bool isAccessibleFrom(Platform p) noexcept;
  Platform getPlatform() noexcept;

 private:
  Platform m_platform;

  std::size_t m_offset;
  void* m_ptr;
  bool m_initialized;

  std::mutex m_mutex;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_HipConstantMemoryResource_HPP
