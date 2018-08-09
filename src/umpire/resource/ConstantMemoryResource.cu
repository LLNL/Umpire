//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by Zifan Nan, nan1@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
// #ifndef UMPIRE_DefaultMemoryResource_INL
// #define UMPIRE_DefaultMemoryResource_INL

#include "umpire/resource/ConstantMemoryResource.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

#include <memory>
#include <sstream>

namespace umpire {
namespace resource {

// template<typename _allocator>
ConstantMemoryResource::ConstantMemoryResource(Platform platform, const std::string& name, int id) :
  MemoryResource(name, id),
  // m_allocator(),
  m_current_size(0l),
  m_highwatermark(0l),
  m_platform(platform)
{
  offset = 0;
  std::cout << "new ConstantMemoryResource" << std::endl;
}

// template<typename _allocator>
void* ConstantMemoryResource::allocate(size_t bytes)
{
  // void* ptr = m_allocator.allocate(bytes);

  void* ptr = nullptr;
  cudaError_t error = ::cudaGetSymbolAddress((void**)&ptr, umpire_internal_device_constant_memory);

  char* new_ptr = (char*)ptr + offset;
  offset += bytes;

  if (offset > 1024 * 64)
  {
    UMPIRE_LOG(Debug, "ask bytes more than max constant memory size (64KB), current size is " << offset - bytes << "bytes");
  }

  ResourceManager::getInstance().registerAllocation((void*)new_ptr, new util::AllocationRecord{ptr, bytes, this->shared_from_this()});

  m_current_size += bytes;
  if (m_current_size > m_highwatermark)
    m_highwatermark = m_current_size;

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);

  return (void*)new_ptr;
}

// template<typename _allocator>
void ConstantMemoryResource::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  // m_allocator.deallocate(ptr);
  util::AllocationRecord* record = ResourceManager::getInstance().deregisterAllocation(ptr);
  m_current_size -= record->m_size;
  delete record;
}

// template<typename _allocator>
long ConstantMemoryResource::getCurrentSize()
{
  UMPIRE_LOG(Debug, "() returning " << m_current_size);
  return m_current_size;
}

// template<typename _allocator>
long ConstantMemoryResource::getHighWatermark()
{
  UMPIRE_LOG(Debug, "() returning " << m_highwatermark);
  return m_highwatermark;
}

// template<typename _allocator>
Platform ConstantMemoryResource::getPlatform()
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
// #endif // UMPIRE_DefaultMemoryResource_INL
