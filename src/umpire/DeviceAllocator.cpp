//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/DeviceAllocator.hpp"
#include "umpire/device_allocator_helper.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/resource/MemoryResourceTypes.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {

__host__ DeviceAllocator::DeviceAllocator(Allocator allocator, size_t size, const char* name, size_t id)
    : m_allocator(allocator),
      m_id(id),
      m_ptr(static_cast<char*>(m_allocator.allocate(size))),
      m_name(name),
      m_size(size),
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
      m_name(other.m_name),
      m_counter(other.m_counter),
      m_size(other.m_size),
      m_child(true)
{
}

__host__ __device__ DeviceAllocator::~DeviceAllocator()
{
}

__host__ void DeviceAllocator::destroy()
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto device_alloc = rm.getAllocator(umpire::resource::Device);

  if(m_counter != nullptr)
    device_alloc.deallocate(m_counter);
  if(m_counter != nullptr)
    m_allocator.deallocate(m_ptr);
}

__device__ void* DeviceAllocator::allocate(size_t size)
{
  std::size_t counter = atomicAdd(m_counter, size);
  if (*m_counter > m_size) {
    UMPIRE_ERROR("DeviceAllocator out of space");
  }

  return static_cast<void*>(m_ptr + counter);
}

__host__ __device__ size_t DeviceAllocator::getID()
{
  return m_id;
}

__device__ const char* DeviceAllocator::getName()
{
  return m_name;
}

} // end of namespace umpire
