// wrapResourceManager.cpp
// This is generated code, do not edit
#include "wrapResourceManager.h"
#include <string>
#include "umpire/ResourceManager.hpp"

// splicer begin class.ResourceManager.CXX_definitions
// splicer end class.ResourceManager.CXX_definitions

extern "C" {

// splicer begin class.ResourceManager.C_definitions
// splicer end class.ResourceManager.C_definitions

umpire_resourcemanager * umpire_resourcemanager_get_instance()
{
// splicer begin class.ResourceManager.method.get_instance
    umpire::ResourceManager & SHCXX_rv = umpire::
        ResourceManager::getInstance();
    umpire_resourcemanager * SHC_rv = 
        static_cast<umpire_resourcemanager *>(static_cast<void *>(
        &SHCXX_rv));
    return SHC_rv;
// splicer end class.ResourceManager.method.get_instance
}

umpire_allocator * umpire_resourcemanager_get_allocator(
    umpire_resourcemanager * self, const char * name)
{
// splicer begin class.ResourceManager.method.get_allocator
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    const std::string SH_name(name);
    umpire::Allocator * SH_rv = new umpire::Allocator(SH_this->getAllocator(SH_name)); 
    umpire_allocator * XSH_rv = static_cast<umpire_allocator *>(static_cast<void *>(SH_rv)); 
    return XSH_rv;

// splicer end class.ResourceManager.method.get_allocator
}

umpire_allocator * umpire_resourcemanager_get_allocator_bufferify(
    umpire_resourcemanager * self, const char * name, int Lname)
{
// splicer begin class.ResourceManager.method.get_allocator_bufferify
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    const std::string SH_name(name, Lname);
    umpire::Allocator * SH_rv = new umpire::Allocator(SH_this->getAllocator(SH_name)); 
    umpire_allocator * XSH_rv = static_cast<umpire_allocator *>(static_cast<void *>(SH_rv)); 
    return XSH_rv;

// splicer end class.ResourceManager.method.get_allocator_bufferify
}

void umpire_resourcemanager_delete_allocator(
    umpire_allocator * alloc_obj)
{
// splicer begin class.ResourceManager.method.delete_allocator
    umpire::Allocator * SHCXX_alloc_obj = static_cast<umpire::
        Allocator *>(static_cast<void *>(alloc_obj));
    delete SHCXX_alloc_obj;

// splicer end class.ResourceManager.method.delete_allocator
}

void umpire_resourcemanager_copy(umpire_resourcemanager * self,
    void * src_ptr, void * dst_ptr)
{
// splicer begin class.ResourceManager.method.copy
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    SH_this->copy(src_ptr, dst_ptr);
    return;
// splicer end class.ResourceManager.method.copy
}

void umpire_resourcemanager_deallocate(umpire_resourcemanager * self,
    void * ptr)
{
// splicer begin class.ResourceManager.method.deallocate
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    SH_this->deallocate(ptr);
    return;
// splicer end class.ResourceManager.method.deallocate
}

}  // extern "C"
