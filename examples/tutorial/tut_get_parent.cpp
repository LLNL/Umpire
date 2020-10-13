//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"

using parent_resource = umpire::MemoryResourceTraits::resource_type;

void check_parent_host(umpire::Allocator alloc)
{
  if(parent_resource::host == alloc.getParentResource()->getTraits().resource)
    std::cout << "The host parent resource is correct" << std::endl;
  else
    std::cout << "Error: Parent resource doesn't match expected type!" << std::endl;
}
void check_parent_device(umpire::Allocator alloc)
{
  if(parent_resource::device == alloc.getParentResource()->getTraits().resource)
    std::cout << "The device parent resource is correct" << std::endl;
  else
    std::cout << "Error: Parent resource doesn't match expected type!" << std::endl;
}
void check_parent_um(umpire::Allocator alloc)
{
  if(parent_resource::um == alloc.getParentResource()->getTraits().resource)
    std::cout << "The um parent resource is correct" << std::endl;
  else
    std::cout << "Error: Parent resource doesn't match expected type!" << std::endl;
}
void check_parent_pinned(umpire::Allocator alloc)
{
  if(parent_resource::pinned == alloc.getParentResource()->getTraits().resource)
    std::cout << "The pinned parent resource is correct" << std::endl;
  else
    std::cout << "Error: Parent resource doesn't match expected type!" << std::endl;
}

void allocate_and_deallocate_pool(const std::string& resource)
{
  constexpr std::size_t SIZE = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator(resource);

  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
      resource + "_pool", allocator);

  double* data =
      static_cast<double*>(pooled_allocator.allocate(SIZE * sizeof(double)));

  std::cout << "Allocated " << (SIZE * sizeof(double)) << " bytes using the "
            << pooled_allocator.getName() << " allocator" << std::endl;

  if(resource == "HOST")
    check_parent_host(pooled_allocator);
  else if (resource == "DEVICE")
    check_parent_device(pooled_allocator);
  else if (resource == "UM")
    check_parent_um(pooled_allocator);
  else if (resource == "PINNED")
    check_parent_pinned(pooled_allocator);
  else
    std::cout << "Error, no match!" << std::endl << std::endl;

  pooled_allocator.deallocate(data);

  std::cout << "Memory deallocated." << std::endl << std::endl;
}

int main(int, char**)
{
  allocate_and_deallocate_pool("HOST");

#if defined(UMPIRE_ENABLE_CUDA)
  allocate_and_deallocate_pool("DEVICE");
  allocate_and_deallocate_pool("UM");
  allocate_and_deallocate_pool("PINNED");
#endif
#if defined(UMPIRE_ENABLE_HIP)
  allocate_and_deallocate_pool("DEVICE");
  allocate_and_deallocate_pool("PINNED");
#endif

  return 0;
}
