// typesUmpire.h
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
// For C users and C++ implementation

#ifndef TYPESUMPIRE_H
#define TYPESUMPIRE_H

#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

struct s_UMP_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_UMP_SHROUD_capsule_data UMP_SHROUD_capsule_data;

struct s_umpire_allocator {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_umpire_allocator umpire_allocator;

struct s_umpire_dynamicpool {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_umpire_dynamicpool umpire_dynamicpool;

struct s_umpire_resourcemanager {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_umpire_resourcemanager umpire_resourcemanager;

struct s_UMP_SHROUD_array {
    UMP_SHROUD_capsule_data cxx;      /* address of C++ memory */
    union {
        const void * cvoidp;
        const char * ccharp;
    } addr;
    size_t len;     /* bytes-per-item or character len of data in cxx */
    size_t size;    /* size of data in cxx */
};
typedef struct s_UMP_SHROUD_array UMP_SHROUD_array;

void umpire_SHROUD_memory_destructor(UMP_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESUMPIRE_H
