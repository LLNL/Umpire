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
struct s_umpire_allocator;
typedef struct s_umpire_allocator umpire_allocator;

// splicer begin class.Allocator.C_declarations
// splicer end class.Allocator.C_declarations

void * umpire_allocator_allocate(umpire_allocator * self, size_t bytes);

void umpire_allocator_deallocate(umpire_allocator * self, void * ptr);

#ifdef __cplusplus
}
#endif

#endif  // WRAPALLOCATOR_H
