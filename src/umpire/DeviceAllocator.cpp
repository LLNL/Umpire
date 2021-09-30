//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/DeviceAllocator.hpp"

#include <stdio.h>
#include <string.h>

#include "umpire/ResourceManager.hpp"
#include "umpire/device_allocator_helper.hpp"
#include "umpire/resource/MemoryResourceTypes.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {

__host__ DeviceAllocator::DeviceAllocator(Allocator allocator, size_t size, const std::string& old_name, size_t id)
    : m_allocator(allocator),
      m_id(id),
      m_ptr(static_cast<char*>(m_allocator.allocate(size))),
      m_size(size),
      m_child(false)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto device_alloc = rm.getAllocator(umpire::resource::Device);

  m_counter = static_cast<unsigned int*>(device_alloc.allocate(sizeof(unsigned int)));
  rm.memset(m_counter, 0);

  // convert the string name to a char name
  const char* name = old_name.c_str();
  memset(m_name, '\0', old_name.length() + 1);
  int i = 0;
  do {
    m_name[i] = name[i];
  } while (name[i++] != 0);
}

__host__ __device__ DeviceAllocator::DeviceAllocator(const DeviceAllocator& other)
    : m_allocator(other.m_allocator),
      m_id(other.m_id),
      m_ptr(other.m_ptr),
      m_counter(other.m_counter),
      m_size(other.m_size),
      m_child(true)
{
  int i = 0;
  do {
    m_name[i] = other.m_name[i];
  } while (other.m_name[i++] != 0);
}

__host__ __device__ DeviceAllocator::~DeviceAllocator()
{
}

__host__ void DeviceAllocator::destroy()
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto device_alloc = rm.getAllocator(umpire::resource::Device);

  if (m_counter != nullptr)
    device_alloc.deallocate(m_counter);
  if (m_ptr != nullptr)
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

__host__ __device__ const char* DeviceAllocator::getName()
{
  return m_name;
}

__host__ __device__ bool DeviceAllocator::isInitialized()
{
  if (m_size > 0) {
    return true;
  }
  return false;
}

} // end of namespace umpire
