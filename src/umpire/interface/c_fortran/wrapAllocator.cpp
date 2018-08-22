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

size_t umpire_allocator_get_size(umpire_allocator * self, void * ptr)
{
// splicer begin class.Allocator.method.get_size
    umpire::Allocator *SH_this = static_cast<umpire::Allocator *>(
        static_cast<void *>(self));
    size_t SHC_rv = SH_this->getSize(ptr);
    return SHC_rv;
// splicer end class.Allocator.method.get_size
}

size_t umpire_allocator_get_high_watermark(umpire_allocator * self)
{
// splicer begin class.Allocator.method.get_high_watermark
    umpire::Allocator *SH_this = static_cast<umpire::Allocator *>(
        static_cast<void *>(self));
    size_t SHC_rv = SH_this->getHighWatermark();
    return SHC_rv;
// splicer end class.Allocator.method.get_high_watermark
}

size_t umpire_allocator_get_current_size(umpire_allocator * self)
{
// splicer begin class.Allocator.method.get_current_size
    umpire::Allocator *SH_this = static_cast<umpire::Allocator *>(
        static_cast<void *>(self));
    size_t SHC_rv = SH_this->getCurrentSize();
    return SHC_rv;
// splicer end class.Allocator.method.get_current_size
}

size_t umpire_allocator_get_id(umpire_allocator * self)
{
// splicer begin class.Allocator.method.get_id
    umpire::Allocator *SH_this = static_cast<umpire::Allocator *>(
        static_cast<void *>(self));
    size_t SHC_rv = SH_this->getId();
    return SHC_rv;
// splicer end class.Allocator.method.get_id
}

}  // extern "C"
