// wrapAllocator.cpp
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
#include "wrapAllocator.h"
#include <cstddef>
#include <string>
#include "umpire/Allocator.hpp"

// splicer begin class.Allocator.CXX_definitions
// splicer end class.Allocator.CXX_definitions

extern "C" {


// helper function
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void umpire_ShroudCopyStringAndFree(UMP_SHROUD_array *data, char *c_var, size_t c_var_len) {
    const char *cxx_var = data->addr.ccharp;
    size_t n = c_var_len;
    if (data->len < n) n = data->len;
    strncpy(c_var, cxx_var, n);
    umpire_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
}

// splicer begin class.Allocator.C_definitions
// splicer end class.Allocator.C_definitions

void * umpire_allocator_allocate(umpire_allocator * self, size_t bytes)
{
// splicer begin class.Allocator.method.allocate
    umpire::Allocator *SH_this =
        static_cast<umpire::Allocator *>(self->addr);
    void * SHC_rv = SH_this->allocate(bytes);
    return SHC_rv;
// splicer end class.Allocator.method.allocate
}

void umpire_allocator_deallocate(umpire_allocator * self, void * ptr)
{
// splicer begin class.Allocator.method.deallocate
    umpire::Allocator *SH_this =
        static_cast<umpire::Allocator *>(self->addr);
    SH_this->deallocate(ptr);
    return;
// splicer end class.Allocator.method.deallocate
}

void umpire_allocator_release(umpire_allocator * self)
{
// splicer begin class.Allocator.method.release
    umpire::Allocator *SH_this =
        static_cast<umpire::Allocator *>(self->addr);
    SH_this->release();
    return;
// splicer end class.Allocator.method.release
}

size_t umpire_allocator_get_size(umpire_allocator * self, void * ptr)
{
// splicer begin class.Allocator.method.get_size
    umpire::Allocator *SH_this =
        static_cast<umpire::Allocator *>(self->addr);
    size_t SHC_rv = SH_this->getSize(ptr);
    return SHC_rv;
// splicer end class.Allocator.method.get_size
}

size_t umpire_allocator_get_high_watermark(umpire_allocator * self)
{
// splicer begin class.Allocator.method.get_high_watermark
    umpire::Allocator *SH_this =
        static_cast<umpire::Allocator *>(self->addr);
    size_t SHC_rv = SH_this->getHighWatermark();
    return SHC_rv;
// splicer end class.Allocator.method.get_high_watermark
}

size_t umpire_allocator_get_current_size(umpire_allocator * self)
{
// splicer begin class.Allocator.method.get_current_size
    umpire::Allocator *SH_this =
        static_cast<umpire::Allocator *>(self->addr);
    size_t SHC_rv = SH_this->getCurrentSize();
    return SHC_rv;
// splicer end class.Allocator.method.get_current_size
}

size_t umpire_allocator_get_actual_size(umpire_allocator * self)
{
// splicer begin class.Allocator.method.get_actual_size
    umpire::Allocator *SH_this =
        static_cast<umpire::Allocator *>(self->addr);
    size_t SHC_rv = SH_this->getActualSize();
    return SHC_rv;
// splicer end class.Allocator.method.get_actual_size
}

const char * umpire_allocator_get_name(umpire_allocator * self)
{
// splicer begin class.Allocator.method.get_name
    umpire::Allocator *SH_this =
        static_cast<umpire::Allocator *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getName();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end class.Allocator.method.get_name
}

void umpire_allocator_get_name_bufferify(umpire_allocator * self,
    UMP_SHROUD_array *DSHF_rv)
{
// splicer begin class.Allocator.method.get_name_bufferify
    umpire::Allocator *SH_this =
        static_cast<umpire::Allocator *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getName();
    DSHF_rv->cxx.addr = static_cast<void *>(const_cast<std::string *>
        (&SHCXX_rv));
    DSHF_rv->cxx.idtor = 0;
    if (SHCXX_rv.empty()) {
        DSHF_rv->addr.ccharp = NULL;
        DSHF_rv->len = 0;
    } else {
        DSHF_rv->addr.ccharp = SHCXX_rv.data();
        DSHF_rv->len = SHCXX_rv.size();
    }
    DSHF_rv->size = 1;
    return;
// splicer end class.Allocator.method.get_name_bufferify
}

size_t umpire_allocator_get_id(umpire_allocator * self)
{
// splicer begin class.Allocator.method.get_id
    umpire::Allocator *SH_this =
        static_cast<umpire::Allocator *>(self->addr);
    size_t SHC_rv = SH_this->getId();
    return SHC_rv;
// splicer end class.Allocator.method.get_id
}

}  // extern "C"
