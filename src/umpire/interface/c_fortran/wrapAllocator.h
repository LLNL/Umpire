//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
// wrapAllocator.h
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
 * \file wrapAllocator.h
 * \brief Shroud generated wrapper for Allocator class
 */
// For C users and C++ implementation

#ifndef WRAPALLOCATOR_H
#define WRAPALLOCATOR_H

#include <stddef.h>
#include "typesUmpire.h"

// splicer begin class.Allocator.CXX_declarations
#ifdef __cplusplus
#include <cstring>
#endif
// splicer end class.Allocator.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.Allocator.C_declarations
// splicer end class.Allocator.C_declarations

void umpire_allocator_delete(umpire_allocator * self);

void * umpire_allocator_allocate(umpire_allocator * self, size_t bytes);

void umpire_allocator_deallocate(umpire_allocator * self, void * ptr);

void umpire_allocator_release(umpire_allocator * self);

size_t umpire_allocator_get_size(umpire_allocator * self, void * ptr);

size_t umpire_allocator_get_high_watermark(umpire_allocator * self);

size_t umpire_allocator_get_current_size(umpire_allocator * self);

size_t umpire_allocator_get_actual_size(umpire_allocator * self);

const char * umpire_allocator_get_name(umpire_allocator * self);

void umpire_allocator_get_name_bufferify(umpire_allocator * self,
    UMP_SHROUD_array *DSHF_rv);

size_t umpire_allocator_get_id(umpire_allocator * self);

#ifdef __cplusplus
}
#endif

#endif  // WRAPALLOCATOR_H
