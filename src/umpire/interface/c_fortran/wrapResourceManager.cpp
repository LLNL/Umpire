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
// wrapResourceManager.cpp
// This is generated code, do not edit
// wrapResourceManager.cpp
#include "wrapResourceManager.h"
#include <string>
#include "umpire/ResourceManager.hpp"

namespace umpire {

// splicer begin class.ResourceManager.CXX_definitions
// splicer end class.ResourceManager.CXX_definitions

extern "C" {

// splicer begin class.ResourceManager.C_definitions
// splicer end class.ResourceManager.C_definitions

UMPIRE_resourcemanager * UMPIRE_resourcemanager_get()
{
// splicer begin class.ResourceManager.method.get
    auto& SH_rv = ResourceManager::getInstance(); return static_cast<UMPIRE_resourcemanager *>(static_cast<void *>(&SH_rv));
// splicer end class.ResourceManager.method.get
}

UMPIRE_allocator * UMPIRE_resourcemanager_get_allocator(UMPIRE_resourcemanager * self, const char * space)
{
// splicer begin class.ResourceManager.method.get_allocator
    ResourceManager *SH_this = static_cast<ResourceManager *>(static_cast<void *>(self));
    const std::string SH_space(space);
    Allocator * SH_rv = new Allocator(SH_this->getAllocator(SH_space));
    UMPIRE_allocator * XSH_rv = static_cast<UMPIRE_allocator *>(static_cast<void *>(SH_rv));
    return XSH_rv;
// splicer end class.ResourceManager.method.get_allocator
}

UMPIRE_allocator * UMPIRE_resourcemanager_get_allocator_bufferify(UMPIRE_resourcemanager * self, const char * space, int Lspace)
{
// splicer begin class.ResourceManager.method.get_allocator_bufferify
    ResourceManager *SH_this = static_cast<ResourceManager *>(static_cast<void *>(self));
    const std::string SH_space(space, Lspace);
    Allocator * SH_rv = new Allocator(SH_this->getAllocator(SH_space));
    UMPIRE_allocator * XSH_rv = static_cast<UMPIRE_allocator *>(static_cast<void *>(SH_rv));
    return XSH_rv;
// splicer end class.ResourceManager.method.get_allocator_bufferify
}

void UMPIRE_resourcemanager_copy(UMPIRE_resourcemanager * self, void * src_ptr, void * dst_ptr)
{
// splicer begin class.ResourceManager.method.copy
    ResourceManager *SH_this = static_cast<ResourceManager *>(static_cast<void *>(self));
    SH_this->copy(src_ptr, dst_ptr);
    return;
// splicer end class.ResourceManager.method.copy
}

void UMPIRE_resourcemanager_deallocate(UMPIRE_resourcemanager * self, void * ptr)
{
// splicer begin class.ResourceManager.method.deallocate
    ResourceManager *SH_this = static_cast<ResourceManager *>(static_cast<void *>(self));
    SH_this->deallocate(ptr);
    return;
// splicer end class.ResourceManager.method.deallocate
}

}  // extern "C"

}  // namespace umpire
