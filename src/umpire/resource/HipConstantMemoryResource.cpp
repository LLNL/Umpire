//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/HipConstantMemoryResource.hpp"

#include <memory>
#include <sstream>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

static constexpr int MAX_CONST_MEM_SIZE = 64 * 1024;
__constant__ static char s_umpire_internal_device_constant_memory[MAX_CONST_MEM_SIZE];

namespace umpire {
namespace resource {

HipConstantMemoryResource::HipConstantMemoryResource(const std::string& name, int id, MemoryResourceTraits traits)
    : MemoryResource{name, id, traits}, m_platform{Platform::hip}, m_offset{0}, m_ptr{nullptr}, m_initialized{false}
{
}

void* HipConstantMemoryResource::allocate(std::size_t bytes)
{
  std::lock_guard<std::mutex> lock{m_mutex};

  if (!m_initialized) {
    hipError_t error = hipGetSymbolAddress((void**)&m_ptr, s_umpire_internal_device_constant_memory);

    if (error != hipSuccess) {
      UMPIRE_ERROR(runtime_error, fmt::format("hipGetSymbolAddress failed with error: {}", hipGetErrorString(error)));
    }

    m_initialized = true;
  }

  char* ptr{static_cast<char*>(m_ptr) + m_offset};
  m_offset += bytes;

  void* ret{static_cast<void*>(ptr)};

  if (m_offset > MAX_CONST_MEM_SIZE) {
    UMPIRE_ERROR(runtime_error, fmt::format("Max total size of constant allocations is 64KB, current size is {} bytes",
                                            (m_offset - bytes)));
  }

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  return ret;
}

void HipConstantMemoryResource::deallocate(void* ptr, std::size_t size)
{
  std::lock_guard<std::mutex> lock{m_mutex};

  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  if ((static_cast<char*>(m_ptr) + (m_offset - size)) == static_cast<char*>(ptr)) {
    m_offset -= size;
  } else {
    UMPIRE_ERROR(runtime_error, "HipConstantMemory deallocations must be in reverse order");
  }
}

bool HipConstantMemoryResource::isAccessibleFrom(Platform p) noexcept
{
  if (p == Platform::hip)
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
