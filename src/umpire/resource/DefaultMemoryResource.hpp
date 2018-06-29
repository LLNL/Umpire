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
#ifndef UMPIRE_DefaultMemoryResource_HPP
#define UMPIRE_DefaultMemoryResource_HPP

#include "umpire/resource/MemoryResource.hpp"

#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace resource {


  /*!
   * \brief Concrete MemoryResource object that uses the template _allocator to
   * allocate and deallocate memory.
   */
template <typename _allocator>
class DefaultMemoryResource :
  public MemoryResource
{
  public: 
    DefaultMemoryResource(Platform platform, const std::string& name, int id);

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    long getCurrentSize();
    long getHighWatermark();

    Platform getPlatform();

  protected: 
    _allocator m_allocator;

    long m_current_size;
    long m_highwatermark;

    Platform m_platform;
};

} // end of namespace resource
} // end of namespace umpire

#include "umpire/resource/DefaultMemoryResource.inl"

#endif // UMPIRE_DefaultMemoryResource_HPP
