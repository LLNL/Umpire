// typesUmpire.h
// This is generated code, do not edit
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

struct s_um_allocator {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_um_allocator um_allocator;

struct s_um_resourcemanager {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_um_resourcemanager um_resourcemanager;

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

void um_SHROUD_memory_destructor(UMP_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESUMPIRE_H
