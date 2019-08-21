//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
// wrapResourceManager.cpp
// This is generated code, do not edit
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
#include "wrapResourceManager.h"
#include <stdlib.h>
#include <string>
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

// splicer begin class.ResourceManager.CXX_definitions
// splicer end class.ResourceManager.CXX_definitions

extern "C" {

// splicer begin class.ResourceManager.C_definitions
// splicer end class.ResourceManager.C_definitions

umpire_resourcemanager * umpire_resourcemanager_get_instance(
    umpire_resourcemanager * SHC_rv)
{
// splicer begin class.ResourceManager.method.get_instance
    umpire::ResourceManager & SHCXX_rv =
        umpire::ResourceManager::getInstance();
    SHC_rv->addr = static_cast<void *>(&SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.ResourceManager.method.get_instance
}

umpire_allocator * umpire_resourcemanager_get_allocator_by_name(
    umpire_resourcemanager * self, const char * name,
    umpire_allocator * SHC_rv)
{
// splicer begin class.ResourceManager.method.get_allocator_by_name
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    umpire::Allocator * SHCXX_rv = new umpire::Allocator;
    const std::string SH_name(name);
    *SHCXX_rv = SH_this->getAllocator(SH_name);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
// splicer end class.ResourceManager.method.get_allocator_by_name
}

umpire_allocator * umpire_resourcemanager_get_allocator_by_name_bufferify(
    umpire_resourcemanager * self, const char * name, int Lname,
    umpire_allocator * SHC_rv)
{
// splicer begin class.ResourceManager.method.get_allocator_by_name_bufferify
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    umpire::Allocator * SHCXX_rv = new umpire::Allocator;
    const std::string SH_name(name, Lname);
    *SHCXX_rv = SH_this->getAllocator(SH_name);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
// splicer end class.ResourceManager.method.get_allocator_by_name_bufferify
}

umpire_allocator * umpire_resourcemanager_get_allocator_by_id(
    umpire_resourcemanager * self, const int id,
    umpire_allocator * SHC_rv)
{
// splicer begin class.ResourceManager.method.get_allocator_by_id
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    umpire::Allocator * SHCXX_rv = new umpire::Allocator;
    *SHCXX_rv = SH_this->getAllocator(id);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
// splicer end class.ResourceManager.method.get_allocator_by_id
}

umpire_allocator * umpire_resourcemanager_get_allocator_for_ptr(
    umpire_resourcemanager * self, void * ptr,
    umpire_allocator * SHC_rv)
{
// splicer begin class.ResourceManager.method.get_allocator_for_ptr
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    umpire::Allocator * SHCXX_rv = new umpire::Allocator;
    *SHCXX_rv = SH_this->getAllocator(ptr);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
// splicer end class.ResourceManager.method.get_allocator_for_ptr
}

void umpire_resourcemanager_copy_all(umpire_resourcemanager * self,
    void * src_ptr, void * dst_ptr)
{
// splicer begin class.ResourceManager.method.copy_all
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    SH_this->copy(src_ptr, dst_ptr);
    return;
// splicer end class.ResourceManager.method.copy_all
}

void umpire_resourcemanager_copy_with_size(
    umpire_resourcemanager * self, void * src_ptr, void * dst_ptr,
    size_t size)
{
// splicer begin class.ResourceManager.method.copy_with_size
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    SH_this->copy(src_ptr, dst_ptr, size);
    return;
// splicer end class.ResourceManager.method.copy_with_size
}

void umpire_resourcemanager_memset_all(umpire_resourcemanager * self,
    void * ptr, int val)
{
// splicer begin class.ResourceManager.method.memset_all
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    SH_this->memset(ptr, val);
    return;
// splicer end class.ResourceManager.method.memset_all
}

void umpire_resourcemanager_memset_with_size(
    umpire_resourcemanager * self, void * ptr, int val, size_t length)
{
// splicer begin class.ResourceManager.method.memset_with_size
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    SH_this->memset(ptr, val, length);
    return;
// splicer end class.ResourceManager.method.memset_with_size
}

void * umpire_resourcemanager_reallocate_default(
    umpire_resourcemanager * self, void * src_ptr, size_t size)
{
// splicer begin class.ResourceManager.method.reallocate_default
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    void * SHC_rv = SH_this->reallocate(src_ptr, size);
    return SHC_rv;
// splicer end class.ResourceManager.method.reallocate_default
}

void * umpire_resourcemanager_reallocate_with_allocator(
    umpire_resourcemanager * self, void * src_ptr, size_t size,
    umpire_allocator allocator)
{
// splicer begin class.ResourceManager.method.reallocate_with_allocator
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    umpire::Allocator * SHCXX_allocator =
        static_cast<umpire::Allocator *>(allocator.addr);
    void * SHC_rv = SH_this->reallocate(src_ptr, size,
        *SHCXX_allocator);
    return SHC_rv;
// splicer end class.ResourceManager.method.reallocate_with_allocator
}

void * umpire_resourcemanager_move(umpire_resourcemanager * self,
    void * src_ptr, umpire_allocator allocator)
{
// splicer begin class.ResourceManager.method.move
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    umpire::Allocator * SHCXX_allocator =
        static_cast<umpire::Allocator *>(allocator.addr);
    void * SHC_rv = SH_this->move(src_ptr, *SHCXX_allocator);
    return SHC_rv;
// splicer end class.ResourceManager.method.move
}

void umpire_resourcemanager_deallocate(umpire_resourcemanager * self,
    void * ptr)
{
// splicer begin class.ResourceManager.method.deallocate
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    SH_this->deallocate(ptr);
    return;
// splicer end class.ResourceManager.method.deallocate
}

size_t umpire_resourcemanager_get_size(umpire_resourcemanager * self,
    void * ptr)
{
// splicer begin class.ResourceManager.method.get_size
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    size_t SHC_rv = SH_this->getSize(ptr);
    return SHC_rv;
// splicer end class.ResourceManager.method.get_size
}

umpire_allocator * umpire_resourcemanager_make_allocator_pool(
    umpire_resourcemanager * self, const char * name,
    umpire_allocator allocator, size_t initial_size, size_t block,
    umpire_allocator * SHC_rv)
{
// splicer begin class.ResourceManager.method.make_allocator_pool
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    umpire::Allocator * SHCXX_rv = new umpire::Allocator;
    const std::string SH_name(name);
    umpire::Allocator * SHCXX_allocator =
        static_cast<umpire::Allocator *>(allocator.addr);
    *SHCXX_rv = SH_this->makeAllocator<umpire::strategy::DynamicPool>(
        SH_name, *SHCXX_allocator, initial_size, block);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
// splicer end class.ResourceManager.method.make_allocator_pool
}

umpire_allocator * umpire_resourcemanager_make_allocator_bufferify_pool(
    umpire_resourcemanager * self, const char * name, int Lname,
    umpire_allocator allocator, size_t initial_size, size_t block,
    umpire_allocator * SHC_rv)
{
// splicer begin class.ResourceManager.method.make_allocator_bufferify_pool
    umpire::ResourceManager *SH_this =
        static_cast<umpire::ResourceManager *>(self->addr);
    umpire::Allocator * SHCXX_rv = new umpire::Allocator;
    const std::string SH_name(name, Lname);
    umpire::Allocator * SHCXX_allocator =
        static_cast<umpire::Allocator *>(allocator.addr);
    *SHCXX_rv = SH_this->makeAllocator<umpire::strategy::DynamicPool>(
        SH_name, *SHCXX_allocator, initial_size, block);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
// splicer end class.ResourceManager.method.make_allocator_bufferify_pool
}

}  // extern "C"
