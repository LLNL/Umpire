// wrapResourceManager.h
// This is generated code, do not edit
/**
 * \file wrapResourceManager.h
 * \brief Shroud generated wrapper for ResourceManager class
 */
// For C users and C++ implementation

#ifndef WRAPRESOURCEMANAGER_H
#define WRAPRESOURCEMANAGER_H

// splicer begin class.ResourceManager.CXX_declarations
// splicer end class.ResourceManager.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// declaration of wrapped types
struct s_UMPIRE_allocator;
typedef struct s_UMPIRE_allocator UMPIRE_allocator;
struct s_UMPIRE_resourcemanager;
typedef struct s_UMPIRE_resourcemanager UMPIRE_resourcemanager;

// splicer begin class.ResourceManager.C_declarations
// splicer end class.ResourceManager.C_declarations

UMPIRE_resourcemanager * UMPIRE_resourcemanager_get();

UMPIRE_allocator * UMPIRE_resourcemanager_get_allocator(UMPIRE_resourcemanager * self, const char * space);

UMPIRE_allocator * UMPIRE_resourcemanager_get_allocator_bufferify(UMPIRE_resourcemanager * self, const char * space, int Lspace);

void UMPIRE_resourcemanager_copy(UMPIRE_resourcemanager * self, void * src_ptr, void * dst_ptr);

void UMPIRE_resourcemanager_deallocate(UMPIRE_resourcemanager * self, void * ptr);

#ifdef __cplusplus
}
#endif

#endif  // WRAPRESOURCEMANAGER_H
