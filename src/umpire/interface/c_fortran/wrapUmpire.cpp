// wrapUmpire.cpp
// This is generated code, do not edit
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
#include "wrapUmpire.h"
#include <cstdlib>
#include "typesUmpire.h"

#include "umpire/Allocator.hpp"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

int umpire_get_invalid_allocator_id()
{
// splicer begin function.get_invalid_allocator_id
    return umpire::invalid_allocator_id;
// splicer end function.get_invalid_allocator_id
}

// Release library allocated memory.
void umpire_SHROUD_memory_destructor(umpire_SHROUD_capsule_data *cap)
{
    void *ptr = cap->addr;
    switch (cap->idtor) {
    case 0:   // --none--
    {
        // Nothing to delete
        break;
    }
    case 1:   // umpire::Allocator
    {
        umpire::Allocator *cxx_ptr = 
            reinterpret_cast<umpire::Allocator *>(ptr);
        delete cxx_ptr;
        break;
    }
    default:
    {
        // Unexpected case in destructor
        break;
    }
    }
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
