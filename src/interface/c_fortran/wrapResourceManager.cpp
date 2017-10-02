// wrapResourceManager.cpp
// This is generated code, do not edit
// wrapResourceManager.cpp
#include "wrapResourceManager.h"
#include <string>
#include "umpire/ResourceManager.hpp"

namespace umpire {

// splicer begin class.ResourceManager.CXX_definitions
// splicer end class.ResourceManager.CXX_definitions

extern "C" {

// splicer begin class.ResourceManager.C_definitions
// splicer end class.ResourceManager.C_definitions

umpire_resourcemanager * umpire_resourcemanager_new()
{
// splicer begin class.ResourceManager.method.new
    ResourceManager * SH_rv = new ResourceManager();
    return static_cast<umpire_resourcemanager *>(static_cast<void *>(SH_rv));
// splicer end class.ResourceManager.method.new
}

void umpire_resourcemanager_delete(umpire_resourcemanager * self)
{
// splicer begin class.ResourceManager.method.delete
    ResourceManager *SH_this = static_cast<ResourceManager *>(static_cast<void *>(self));
    delete SH_this;
    return;
// splicer end class.ResourceManager.method.delete
}

umpire_allocator umpire_resourcemanager_get_allocator(umpire_resourcemanager * self, const char * space)
{
// splicer begin class.ResourceManager.method.get_allocator
    ResourceManager *SH_this = static_cast<ResourceManager *>(static_cast<void *>(self));
    const std::string SH_space(space);
    Allocator SH_rv = SH_this->getAllocator(SH_space);
    umpire_allocator XSH_rv = static_cast<umpire_allocator *>(static_cast<void *>(SH_rv));
    return XSH_rv;
// splicer end class.ResourceManager.method.get_allocator
}

umpire_allocator umpire_resourcemanager_get_allocator_bufferify(umpire_resourcemanager * self, const char * space, int Lspace)
{
// splicer begin class.ResourceManager.method.get_allocator_bufferify
    ResourceManager *SH_this = static_cast<ResourceManager *>(static_cast<void *>(self));
    const std::string SH_space(space, Lspace);
    Allocator SH_rv = SH_this->getAllocator(SH_space);
    umpire_allocator XSH_rv = static_cast<umpire_allocator *>(static_cast<void *>(SH_rv));
    return XSH_rv;
// splicer end class.ResourceManager.method.get_allocator_bufferify
}

}  // extern "C"

}  // namespace umpire
