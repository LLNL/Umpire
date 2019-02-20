//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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

#include "umpire/resource/CudaConstantMemoryResource.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

#include <memory>
#include <sstream>

namespace umpire {
namespace resource {

CudaConstantMemoryResource::CudaConstantMemoryResource(const std::string& name, int id, MemoryResourceTraits traits) :
  MemoryResource(name, id, traits),
  umpire::strategy::mixins::Inspector(),
  m_platform(Platform::cuda),
  m_offset(0),
  m_ptr(nullptr)
{
  cudaError_t error = ::cudaGetSymbolAddress((void**)&m_ptr, umpire_internal_device_constant_memory);

  if (error != cudaSuccess) {
    UMPIRE_ERROR("cudaGetSymbolAddress failed with error: " << cudaGetErrorString(error));
  }
}

void* CudaConstantMemoryResource::allocate(size_t bytes)
{
  char* ptr = static_cast<char*>(m_ptr) + m_offset;
  m_offset += bytes;

  void* ret = static_cast<void*>(ptr);

  if (m_offset > 1024 * 64)
  {
    UMPIRE_ERROR("Max total size of constant allocations is 64KB, current size is " << m_offset - bytes << "bytes");
  }

  registerAllocation(ret, bytes, this->shared_from_this());

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  return ret;
}

void CudaConstantMemoryResource::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  auto prev_size = m_current_size;
  deregisterAllocation(ptr);
  auto record_size = prev_size - m_current_size;

  if ( (static_cast<char*>(m_ptr) + (m_offset - record_size))
      == static_cast<char*>(ptr)) {
    m_offset -= record_size;
  } else {
    UMPIRE_ERROR("CudaConstantMemory deallocations must be in reverse order");
  }
}

long CudaConstantMemoryResource::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_current_size);
  return m_current_size;
}

long CudaConstantMemoryResource::getHighWatermark() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_high_watermark);
  return m_high_watermark;
}

Platform CudaConstantMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
