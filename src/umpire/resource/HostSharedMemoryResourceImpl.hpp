//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>

#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

class HostSharedMemoryResource::impl {
  public:
    impl(const std::string& name, std::size_t /* size */) :
      m_segment_name{ name }
    {
    }

    ~impl()
    {
    }

    void* allocate(const std::string& /* name */, std::size_t /* bytes */)
    {
      return static_cast<void*>(nullptr);
    }

    void deallocate(void* /* ptr */)
    {
    }

    void* find_pointer_from_name(std::string /* name */)
    {
      return nullptr;
    }

    std::size_t getCurrentSize() const noexcept;
    std::size_t getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;

  private:
    std::string m_segment_name;
    std::size_t m_current_size{0};
    std::size_t m_high_watermark{0};
};

} // end of namespace resource
} // end of namespace umpire
