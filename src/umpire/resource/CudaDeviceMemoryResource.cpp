//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/CudaDeviceMemoryResource.hpp"

#include <memory>
#include <sstream>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

CudaDeviceMemoryResource::CudaDeviceMemoryResource(Platform platform,
                                                   const std::string& name,
                                                   int id,
                                                   MemoryResourceTraits traits)
    : MemoryResource(name, id, traits), m_allocator{}, m_platform(platform)
{
}

void* CudaDeviceMemoryResource::allocate(std::size_t bytes)
{
  int old_device;
  cudaGetDevice(&old_device);
  if (old_device != m_traits.id)
    cudaSetDevice(m_traits.id);

  void* ptr = m_allocator.allocate(bytes);

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);

  if (old_device != m_traits.id)
    cudaSetDevice(old_device);
  return ptr;
}

void CudaDeviceMemoryResource::deallocate(void* ptr)
{
  int old_device;
  cudaGetDevice(&old_device);
  if (old_device != m_traits.id)
    cudaSetDevice(m_traits.id);

  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  m_allocator.deallocate(ptr);
  if (old_device != m_traits.id)
    cudaSetDevice(old_device);
}

std::size_t CudaDeviceMemoryResource::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << 0);
  return 0;
}

std::size_t CudaDeviceMemoryResource::getHighWatermark() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << 0);
  return 0;
}

bool CudaDeviceMemoryResource::isAccessibleFrom(Platform p) noexcept
{
  if(p == Platform::undefined || p == Platform::host)
    return false;
  else
    return true;
}

Platform CudaDeviceMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
