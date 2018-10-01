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

#include "umpire/resource/ConstantMemoryResource.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

#include <memory>
#include <sstream>

namespace umpire {
namespace resource {

ConstantMemoryResource::ConstantMemoryResource(const std::string& name, int id) :
  MemoryResource(name, id),
  m_current_size(0l),
  m_highwatermark(0l),
  m_platform(Platform::cuda),
  m_offset(0),
  m_ptr(nullptr)
{
  cudaError_t error = ::cudaGetSymbolAddress((void**)&m_ptr, umpire_internal_device_constant_memory);
}

void* ConstantMemoryResource::allocate(size_t bytes)
{
  char* ptr = static_cast<char*>(m_ptr) + m_offset;
  m_offset += bytes;

  void* ret = static_cast<void*>(ptr);

  if (m_offset > 1024 * 64)
  {
    UMPIRE_ERROR("Max total size of constant allocations is 64KB, current size is " << m_offset - bytes << "bytes");
  }

  ResourceManager::getInstance().registerAllocation(
      ret, new util::AllocationRecord{ret, bytes, this->shared_from_this()});

  m_current_size += bytes;
  if (m_current_size > m_highwatermark)
    m_highwatermark = m_current_size;

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  return ret;
}

void ConstantMemoryResource::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  util::AllocationRecord* record = ResourceManager::getInstance().deregisterAllocation(ptr);
  m_current_size -= record->m_size;

  if ( (static_cast<char*>(m_ptr) + (m_offset - record->m_size)) 
      == static_cast<char*>(ptr)) {
    m_offset -= record->m_size;
  } else {
    UMPIRE_ERROR("ConstantMemory deallocations must be in reverse order");
  }

  delete record;
}

long ConstantMemoryResource::getCurrentSize()
{
  UMPIRE_LOG(Debug, "() returning " << m_current_size);
  return m_current_size;
}

long ConstantMemoryResource::getHighWatermark()
{
  UMPIRE_LOG(Debug, "() returning " << m_highwatermark);
  return m_highwatermark;
}

Platform ConstantMemoryResource::getPlatform()
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
