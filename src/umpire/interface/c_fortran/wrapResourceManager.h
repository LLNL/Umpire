// wrapResourceManager.h
// This is generated code, do not edit
/**
 * \file wrapResourceManager.h
 * \brief Shroud generated wrapper for ResourceManager class
 */
// For C users and C++ implementation

#ifndef WRAPRESOURCEMANAGER_H
#define WRAPRESOURCEMANAGER_H

#include <stddef.h>

// splicer begin class.ResourceManager.CXX_declarations
// splicer end class.ResourceManager.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// declaration of shadow types
struct s_umpire_allocator;
typedef struct s_umpire_allocator umpire_allocator;
struct s_umpire_resourcemanager;
typedef struct s_umpire_resourcemanager umpire_resourcemanager;

// splicer begin class.ResourceManager.C_declarations
// splicer end class.ResourceManager.C_declarations

umpire_resourcemanager * umpire_resourcemanager_getinstance();

umpire_allocator * umpire_resourcemanager_get_allocator_0(
    umpire_resourcemanager * self, const char * name);

umpire_allocator * umpire_resourcemanager_get_allocator_0_bufferify(
    umpire_resourcemanager * self, const char * name, int Lname);

umpire_allocator * umpire_resourcemanager_get_allocator_1(
    umpire_resourcemanager * self, const int id);

void umpire_resourcemanager_delete_allocator(
    umpire_allocator * alloc_obj);

void umpire_resourcemanager_copy_0(umpire_resourcemanager * self,
    void * src_ptr, void * dst_ptr);

void umpire_resourcemanager_copy_1(umpire_resourcemanager * self,
    void * src_ptr, void * dst_ptr, size_t size);

void umpire_resourcemanager_memset_0(umpire_resourcemanager * self,
    void * ptr, int val);

void umpire_resourcemanager_memset_1(umpire_resourcemanager * self,
    void * ptr, int val, size_t length);

void * umpire_resourcemanager_reallocate(umpire_resourcemanager * self,
    void * src_ptr, size_t size);

void umpire_resourcemanager_deallocate(umpire_resourcemanager * self,
    void * ptr);

size_t umpire_resourcemanager_get_size(umpire_resourcemanager * self,
    void * ptr);

#ifdef __cplusplus
}
#endif

#endif  // WRAPRESOURCEMANAGER_H
