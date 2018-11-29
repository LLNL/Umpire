// wrapAllocator.h
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
 * \file wrapAllocator.h
 * \brief Shroud generated wrapper for Allocator class
 */
// For C users and C++ implementation

#ifndef WRAPALLOCATOR_H
#define WRAPALLOCATOR_H

#include <stddef.h>
#include "typesUmpire.h"

// splicer begin class.Allocator.CXX_declarations
#include <cstring>
// splicer end class.Allocator.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.Allocator.C_declarations
// splicer end class.Allocator.C_declarations

void * um_allocator_allocate(um_allocator * self, size_t bytes);

void um_allocator_deallocate(um_allocator * self, void * ptr);

size_t um_allocator_get_size(um_allocator * self, void * ptr);

size_t um_allocator_get_high_watermark(um_allocator * self);

size_t um_allocator_get_current_size(um_allocator * self);

void um_allocator_get_name_bufferify(um_allocator * self,
    UMP_SHROUD_array *DSHF_rv);

size_t um_allocator_get_id(um_allocator * self);

#ifdef __cplusplus
}
#endif

#endif  // WRAPALLOCATOR_H
