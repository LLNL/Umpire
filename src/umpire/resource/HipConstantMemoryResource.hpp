//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipConstantMemoryResource_HPP
#define UMPIRE_HipConstantMemoryResource_HPP

#include <hip/hip_runtime.h>

#include <mutex>

#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace resource {

class HipConstantMemoryResource : public MemoryResource {
 public:
  HipConstantMemoryResource(const std::string& name, int id,
                            MemoryResourceTraits traits);

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

  std::mutex m_mutex;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_HipConstantMemoryResource_HPP
