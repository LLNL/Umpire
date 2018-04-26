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

// declaration of shadow types
struct s_umpire_allocator;
typedef struct s_umpire_allocator umpire_allocator;
struct s_umpire_resourcemanager;
typedef struct s_umpire_resourcemanager umpire_resourcemanager;

// splicer begin class.ResourceManager.C_declarations
// splicer end class.ResourceManager.C_declarations

umpire_resourcemanager * umpire_resourcemanager_get_instance();

umpire_allocator * umpire_resourcemanager_get_allocator(
    umpire_resourcemanager * self, const char * name);

umpire_allocator * umpire_resourcemanager_get_allocator_bufferify(
    umpire_resourcemanager * self, const char * name, int Lname);

void umpire_resourcemanager_delete_allocator(
    umpire_allocator * alloc_obj);

void umpire_resourcemanager_copy(umpire_resourcemanager * self,
    void * src_ptr, void * dst_ptr);

void umpire_resourcemanager_deallocate(umpire_resourcemanager * self,
    void * ptr);

#ifdef __cplusplus
}
#endif

#endif  // WRAPRESOURCEMANAGER_H
