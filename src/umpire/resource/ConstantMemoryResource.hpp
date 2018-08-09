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
#ifndef UMPIRE_ConstantMemoryResource_HPP
#define UMPIRE_ConstantMemoryResource_HPP

#include "umpire/resource/MemoryResource.hpp"

#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Platform.hpp"

#include <cuda_runtime_api.h>

__constant__ char umpire_internal_device_constant_memory[64*1024];

namespace umpire {
namespace resource {


  /*!
   * \brief Concrete MemoryResource object that uses the template _allocator to
   * allocate and deallocate memory.
   */
class ConstantMemoryResource :
  public MemoryResource
{
  public: 
    ConstantMemoryResource(Platform platform, const std::string& name, int id);

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    long getCurrentSize();
    long getHighWatermark();

    Platform getPlatform();

  protected: 
    // _allocator m_allocator;

    long m_current_size;
    long m_highwatermark;

    Platform m_platform;
  private:
    size_t offset;
};

} // end of namespace resource
} // end of namespace umpire

// #include "umpire/resource/ConstantMemoryResource.inl"

#endif // UMPIRE_DefaultMemoryResource_HPP
