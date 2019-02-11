// wrapResourceManager.h
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
/**
 * \file wrapResourceManager.h
 * \brief Shroud generated wrapper for ResourceManager class
 */
// For C users and C++ implementation

#ifndef WRAPRESOURCEMANAGER_H
#define WRAPRESOURCEMANAGER_H

#include <stddef.h>
#include "typesUmpire.h"

// splicer begin class.ResourceManager.CXX_declarations
// splicer end class.ResourceManager.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.ResourceManager.C_declarations
// splicer end class.ResourceManager.C_declarations

umpire_resourcemanager * umpire_resourcemanager_get_instance(
    umpire_resourcemanager * SHC_rv);

umpire_allocator * umpire_resourcemanager_get_allocator_by_name(
    umpire_resourcemanager * self, const char * name,
    umpire_allocator * SHC_rv);

umpire_allocator * umpire_resourcemanager_get_allocator_by_name_bufferify(
    umpire_resourcemanager * self, const char * name, int Lname,
    umpire_allocator * SHC_rv);

umpire_allocator * umpire_resourcemanager_get_allocator_by_id(
    umpire_resourcemanager * self, const int id,
    umpire_allocator * SHC_rv);

umpire_allocator * umpire_resourcemanager_get_allocatorfor_ptr(
    umpire_resourcemanager * self, void * ptr,
    umpire_allocator * SHC_rv);

void umpire_resourcemanager_copy_all(umpire_resourcemanager * self,
    void * src_ptr, void * dst_ptr);

void umpire_resourcemanager_copy_with_size(
    umpire_resourcemanager * self, void * src_ptr, void * dst_ptr,
    size_t size);

void umpire_resourcemanager_memset_all(umpire_resourcemanager * self,
    void * ptr, int val);

void umpire_resourcemanager_memset_with_size(
    umpire_resourcemanager * self, void * ptr, int val, size_t length);

void * umpire_resourcemanager_reallocate_0(
    umpire_resourcemanager * self, void * src_ptr, size_t size);

void * umpire_resourcemanager_reallocate_with_allocator(
    umpire_resourcemanager * self, void * src_ptr, size_t size,
    umpire_allocator allocator);

void * umpire_resourcemanager_move(umpire_resourcemanager * self,
    void * src_ptr, umpire_allocator allocator);

void umpire_resourcemanager_deallocate(umpire_resourcemanager * self,
    void * ptr);

size_t umpire_resourcemanager_get_size(umpire_resourcemanager * self,
    void * ptr);

umpire_allocator * umpire_resourcemanager_make_allocator_umpire_strategy_DynamicPool(
    umpire_resourcemanager * self, const char * name, int initial_size,
    int block, umpire_allocator * SHC_rv);

umpire_allocator * umpire_resourcemanager_make_allocator_umpire_strategy_DynamicPool_bufferify(
    umpire_resourcemanager * self, const char * name, int Lname,
    int initial_size, int block, umpire_allocator * SHC_rv);

#ifdef __cplusplus
}
#endif

#endif  // WRAPRESOURCEMANAGER_H
