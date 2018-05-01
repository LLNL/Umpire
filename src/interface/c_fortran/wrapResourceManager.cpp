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

umpire_resourcemanager * umpire_resourcemanager_getinstance()
{
// splicer begin class.ResourceManager.method.getinstance
    umpire::ResourceManager & SHCXX_rv = umpire::ResourceManager::getInstance();
    umpire_resourcemanager * SHC_rv = static_cast<umpire_resourcemanager *>(static_cast<void *>(&SHCXX_rv));
    return SHC_rv;

// splicer end class.ResourceManager.method.getinstance
}

void umpire_resourcemanager_initialize(umpire_resourcemanager * self)
{
// splicer begin class.ResourceManager.method.initialize
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    SH_this->initialize();
    return;
// splicer end class.ResourceManager.method.initialize
}

umpire_allocator * umpire_resourcemanager_get_allocator_0(
    umpire_resourcemanager * self, const char * name)
{
// splicer begin class.ResourceManager.method.get_allocator_0
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    const std::string SH_name(name);
    umpire::Allocator * SH_rv = new umpire::Allocator(SH_this->getAllocator(SH_name)); 
    umpire_allocator * XSH_rv = static_cast<umpire_allocator *>(static_cast<void *>(SH_rv)); 
    return XSH_rv;

// splicer end class.ResourceManager.method.get_allocator_0
}

umpire_allocator * umpire_resourcemanager_get_allocator_0_bufferify(
    umpire_resourcemanager * self, const char * name, int Lname)
{
// splicer begin class.ResourceManager.method.get_allocator_0_bufferify
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    const std::string SH_name(name, Lname);
    umpire::Allocator * SH_rv = new umpire::Allocator(SH_this->getAllocator(SH_name)); 
    umpire_allocator * XSH_rv = static_cast<umpire_allocator *>(static_cast<void *>(SH_rv)); 
    return XSH_rv;

// splicer end class.ResourceManager.method.get_allocator_0_bufferify
}

umpire_allocator * umpire_resourcemanager_get_allocator_1(
    umpire_resourcemanager * self, const int id)
{
// splicer begin class.ResourceManager.method.get_allocator_1
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    umpire::Allocator * SH_rv = new umpire::Allocator(SH_this->getAllocator(id)); 
    umpire_allocator * XSH_rv = static_cast<umpire_allocator *>(static_cast<void *>(SH_rv)); 
    return XSH_rv;

// splicer end class.ResourceManager.method.get_allocator_1
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

void umpire_resourcemanager_copy_0(umpire_resourcemanager * self,
    void * src_ptr, void * dst_ptr)
{
// splicer begin class.ResourceManager.method.copy_0
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    SH_this->copy(src_ptr, dst_ptr);
    return;
// splicer end class.ResourceManager.method.copy_0
}

void umpire_resourcemanager_copy_1(umpire_resourcemanager * self,
    void * src_ptr, void * dst_ptr, size_t size)
{
// splicer begin class.ResourceManager.method.copy_1
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    SH_this->copy(src_ptr, dst_ptr, size);
    return;
// splicer end class.ResourceManager.method.copy_1
}

void umpire_resourcemanager_memset_0(umpire_resourcemanager * self,
    void * ptr, int val)
{
// splicer begin class.ResourceManager.method.memset_0
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    SH_this->memset(ptr, val);
    return;
// splicer end class.ResourceManager.method.memset_0
}

void umpire_resourcemanager_memset_1(umpire_resourcemanager * self,
    void * ptr, int val, size_t length)
{
// splicer begin class.ResourceManager.method.memset_1
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    SH_this->memset(ptr, val, length);
    return;
// splicer end class.ResourceManager.method.memset_1
}

void * umpire_resourcemanager_reallocate(umpire_resourcemanager * self,
    void * src_ptr, size_t size)
{
// splicer begin class.ResourceManager.method.reallocate
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    void * SHC_rv = SH_this->reallocate(src_ptr, size);
    return SHC_rv;
// splicer end class.ResourceManager.method.reallocate
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

size_t umpire_resourcemanager_get_size(umpire_resourcemanager * self,
    void * ptr)
{
// splicer begin class.ResourceManager.method.get_size
    umpire::ResourceManager *SH_this = static_cast<umpire::
        ResourceManager *>(static_cast<void *>(self));
    size_t SHC_rv = SH_this->getSize(ptr);
    return SHC_rv;
// splicer end class.ResourceManager.method.get_size
}

}  // extern "C"
