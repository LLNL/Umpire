//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/DeviceAllocator.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/resource/MemoryResourceTypes.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {

__host__ DeviceAllocator::DeviceAllocator(Allocator allocator, size_t size, size_t id)
    : m_allocator(allocator),
      m_id(id),
      m_ptr(static_cast<char*>(m_allocator.allocate(size))),
      m_size(size),
      m_bytes_used(0),
      m_initialized(false),
      m_child(false)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto device_alloc = rm.getAllocator(umpire::resource::Device);

  m_counter = static_cast<unsigned int*>(device_alloc.allocate(sizeof(unsigned int)));
  rm.memset(m_counter, 0);
}

__host__ __device__ DeviceAllocator::DeviceAllocator(const DeviceAllocator& other)
    : m_allocator(other.m_allocator),
      m_id(other.m_id),
      m_ptr(other.m_ptr),
      m_counter(other.m_counter),
      m_size(other.m_size),
      m_bytes_used(other.m_bytes_used),
      m_child(true)
{
}

__host__ __device__ DeviceAllocator::~DeviceAllocator()
{
/*#if !defined(__CUDA_ARCH__)
  if (!m_child) {
    auto& rm = umpire::ResourceManager::getInstance();
    auto device_alloc = rm.getAllocator(umpire::resource::Device);

    device_alloc.deallocate(m_counter);
    m_allocator.deallocate(m_ptr);
  }
#endif*/
}

__device__ void* DeviceAllocator::allocate(size_t size)
{
  std::size_t counter = atomicAdd(m_counter, size);
  if (*m_counter > m_size) {
    UMPIRE_ERROR("DeviceAllocator out of space");
  }

  m_bytes_used += (size);
  return static_cast<void*>(m_ptr + counter);
}

__host__ __device__ size_t DeviceAllocator::getID()
{
  return m_id;
}

__host__ __device__ size_t DeviceAllocator::getBytesUsed()
{
  return m_bytes_used;
}

} // end of namespace umpire
