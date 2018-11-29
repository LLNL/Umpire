// wrapResourceManager.h
// This is generated code, do not edit
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

um_resourcemanager * um_resourcemanager_get_instance(
    um_resourcemanager * SHC_rv);

um_allocator * um_resourcemanager_get_allocator_by_name(
    um_resourcemanager * self, const char * name,
    um_allocator * SHC_rv);

um_allocator * um_resourcemanager_get_allocator_by_name_bufferify(
    um_resourcemanager * self, const char * name, int Lname,
    um_allocator * SHC_rv);

um_allocator * um_resourcemanager_get_allocator_by_id(
    um_resourcemanager * self, const int id, um_allocator * SHC_rv);

um_allocator * um_resourcemanager_get_allocatorfor_ptr(
    um_resourcemanager * self, void * ptr, um_allocator * SHC_rv);

void um_resourcemanager_copy_all(um_resourcemanager * self,
    void * src_ptr, void * dst_ptr);

void um_resourcemanager_copy_with_size(um_resourcemanager * self,
    void * src_ptr, void * dst_ptr, size_t size);

void um_resourcemanager_memset_0(um_resourcemanager * self, void * ptr,
    int val);

void um_resourcemanager_memset_1(um_resourcemanager * self, void * ptr,
    int val, size_t length);

void * um_resourcemanager_reallocate(um_resourcemanager * self,
    void * src_ptr, size_t size);

void um_resourcemanager_deallocate(um_resourcemanager * self,
    void * ptr);

size_t um_resourcemanager_get_size(um_resourcemanager * self,
    void * ptr);

um_allocator * um_resourcemanager_make_allocator_int(
    um_resourcemanager * self, const char * name,
    um_allocator * SHC_rv);

um_allocator * um_resourcemanager_make_allocator_int_bufferify(
    um_resourcemanager * self, const char * name, int Lname,
    um_allocator * SHC_rv);

#ifdef __cplusplus
}
#endif

#endif  // WRAPRESOURCEMANAGER_H
