//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include "umpire/DeviceAllocator.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/resource/MemoryResourceTypes.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {

  __host__
DeviceAllocator::DeviceAllocator(Allocator allocator, size_t size) :
  m_allocator(allocator),
  m_ptr(static_cast<char*>(m_allocator.allocate(size))),
  m_size(size),
  m_child(false)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto device_alloc = rm.getAllocator(umpire::resource::Device);

  m_counter = static_cast<unsigned int*>(
    device_alloc.allocate(sizeof(unsigned int)));
  rm.memset(m_counter, 0);
}

__host__ __device__
DeviceAllocator::DeviceAllocator(const DeviceAllocator& other) :
  m_allocator(other.m_allocator),
  m_ptr(other.m_ptr),
  m_counter(other.m_counter),
  m_size(other.m_size),
  m_child(true)
{
}

__host__
DeviceAllocator::~DeviceAllocator()
{
  if (!m_child) {
    auto& rm = umpire::ResourceManager::getInstance();
    auto device_alloc = rm.getAllocator(umpire::resource::Device);

    device_alloc.deallocate(m_counter);
    m_allocator.deallocate(m_ptr);
  }
}

__device__
void*
DeviceAllocator::allocate(size_t size)
{
  return static_cast<void*>(
    m_ptr + atomicAdd(m_counter, size));
}

} // end of namespace umpire
