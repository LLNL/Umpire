//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/HipConstantMemoryResource.hpp"

#include <memory>
#include <sstream>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

__constant__ static char s_umpire_internal_device_constant_memory[64 * 1024];

namespace umpire {
namespace resource {

HipConstantMemoryResource::HipConstantMemoryResource(
    const std::string& name, int id, MemoryResourceTraits traits)
    : MemoryResource{name, id, traits},
      m_current_size{0},
      m_highwatermark{0},
      m_platform{Platform::hip},
      m_offset{0},
      m_ptr{s_umpire_internal_device_constant_memory}
{
}

void* HipConstantMemoryResource::allocate(std::size_t bytes)
{
  std::lock_guard<std::mutex> lock{m_mutex};

  char* ptr{static_cast<char*>(m_ptr) + m_offset};
  m_offset += bytes;

  void* ret{static_cast<void*>(ptr)};

  if (m_offset > 1024 * 64) {
    UMPIRE_ERROR(
        "Max total size of constant allocations is 64KB, current size is "
        << m_offset - bytes << "bytes");
  }

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  return ret;
}

void HipConstantMemoryResource::deallocate(void* ptr, std::size_t size)
{
  std::lock_guard<std::mutex> lock{m_mutex};

  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  if ((static_cast<char*>(m_ptr) + (m_offset - size)) ==
      static_cast<char*>(ptr)) {
    m_offset -= size;
  } else {
    UMPIRE_ERROR("HipConstantMemory deallocations must be in reverse order");
  }
}

std::size_t HipConstantMemoryResource::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_current_size);
  return m_current_size;
}

std::size_t HipConstantMemoryResource::getHighWatermark() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_highwatermark);
  return m_highwatermark;
}

bool HipConstantMemoryResource::isAccessibleFrom(Platform p) noexcept
{
  if(p == Platform::hip)
    return true;
  else
    return false;
}

Platform HipConstantMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
