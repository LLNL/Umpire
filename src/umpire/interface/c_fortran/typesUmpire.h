//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef TYPESUMPIRE_H
#define TYPESUMPIRE_H

#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

struct s_umpire_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_umpire_SHROUD_capsule_data umpire_SHROUD_capsule_data;

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

struct s_umpire_SHROUD_array {
    umpire_SHROUD_capsule_data cxx;      /* address of C++ memory */
    union {
        const void * cvoidp;
        const char * ccharp;
    } addr;
    size_t len;     /* bytes-per-item or character len of data in cxx */
    size_t size;    /* size of data in cxx */
};
typedef struct s_umpire_SHROUD_array umpire_SHROUD_array;

void umpire_SHROUD_memory_destructor(umpire_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESUMPIRE_H
