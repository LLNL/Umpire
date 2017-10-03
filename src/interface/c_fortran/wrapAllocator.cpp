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

int * UMPIRE_allocator_allocate_int(UMPIRE_allocator * self, size_t bytes)
{
// splicer begin class.Allocator.method.allocate_int
    Allocator *SH_this = static_cast<Allocator *>(static_cast<void *>(self));
    int * SH_rv = static_cast<int*>(SH_this->allocate(bytes*sizeof(int)));
    return SH_rv;
// splicer end class.Allocator.method.allocate_int
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
