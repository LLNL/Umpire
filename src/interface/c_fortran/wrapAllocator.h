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
// wrapAllocator.h
// This is generated code, do not edit
/**
 * \file wrapAllocator.h
 * \brief Shroud generated wrapper for Allocator class
 */
// For C users and C++ implementation

#ifndef WRAPALLOCATOR_H
#define WRAPALLOCATOR_H

#include "stdlib.h"

// splicer begin class.Allocator.CXX_declarations
// splicer end class.Allocator.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// declaration of wrapped types
struct s_UMPIRE_allocator;
typedef struct s_UMPIRE_allocator UMPIRE_allocator;

// splicer begin class.Allocator.C_declarations
// splicer end class.Allocator.C_declarations

void * UMPIRE_allocator_allocate(UMPIRE_allocator * self, size_t bytes);

void UMPIRE_allocator_deallocate(UMPIRE_allocator * self, void * ptr);

#ifdef __cplusplus
}
#endif

#endif  // WRAPALLOCATOR_H
