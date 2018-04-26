// wrapAllocator.cpp
// This is generated code, do not edit
#include "wrapAllocator.h"
#include "umpire/Allocator.hpp"

// splicer begin class.Allocator.CXX_definitions
// splicer end class.Allocator.CXX_definitions

extern "C" {

// splicer begin class.Allocator.C_definitions
// splicer end class.Allocator.C_definitions

void * umpire_allocator_allocate(umpire_allocator * self, size_t bytes)
{
// splicer begin class.Allocator.method.allocate
    umpire::Allocator *SH_this = static_cast<umpire::Allocator *>(
        static_cast<void *>(self));
    void * SHC_rv = SH_this->allocate(bytes);
    return SHC_rv;
// splicer end class.Allocator.method.allocate
}

void umpire_allocator_deallocate(umpire_allocator * self, void * ptr)
{
// splicer begin class.Allocator.method.deallocate
    umpire::Allocator *SH_this = static_cast<umpire::Allocator *>(
        static_cast<void *>(self));
    SH_this->deallocate(ptr);
    return;
// splicer end class.Allocator.method.deallocate
}

}  // extern "C"
