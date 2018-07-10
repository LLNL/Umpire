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
// wrapAllocator.cpp
// This is generated code, do not edit
// wrapAllocator.cpp
#include "wrapAllocator.h"
#include "umpire/Allocator.hpp"

namespace umpire {

// splicer begin class.Allocator.CXX_definitions
// splicer end class.Allocator.CXX_definitions

extern "C" {

// splicer begin class.Allocator.C_definitions
// splicer end class.Allocator.C_definitions

void * UMPIRE_allocator_allocate(UMPIRE_allocator * self, size_t bytes)
{
// splicer begin class.Allocator.method.allocate
    Allocator *SH_this = static_cast<Allocator *>(static_cast<void *>(self));
    void * SH_rv = SH_this->allocate(bytes);
    return SH_rv;
// splicer end class.Allocator.method.allocate
}

void UMPIRE_allocator_deallocate(UMPIRE_allocator * self, void * ptr)
{
// splicer begin class.Allocator.method.deallocate
    Allocator *SH_this = static_cast<Allocator *>(static_cast<void *>(self));
    SH_this->deallocate(ptr);
    return;
// splicer end class.Allocator.method.deallocate
}

}  // extern "C"

}  // namespace umpire
