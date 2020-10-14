//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/QuickPool.hpp"

using parent_resource = umpire::MemoryResourceTraits::resource_type;

void check_parent(umpire::Allocator alloc, const std::string& resource)
{
  if(resource == "HOST") {
    UMPIRE_ASSERT(parent_resource::host == alloc.getParentResource()->getTraits().resource);
    std::cout << "Check passed..." << std::endl;
  } else if (resource == "DEVICE") {
    UMPIRE_ASSERT(parent_resource::device == alloc.getParentResource()->getTraits().resource);
    std::cout << "Check passed..." << std::endl;
  } else if (resource == "UM") {
    UMPIRE_ASSERT(parent_resource::um == alloc.getParentResource()->getTraits().resource);
    std::cout << "Check passed..." << std::endl;
  } else if (resource == "PINNED") {
    UMPIRE_ASSERT(parent_resource::pinned == alloc.getParentResource()->getTraits().resource);
    std::cout << "Check passed..." << std::endl;
  } else {
    std::cout << "Error, no match!" << std::endl << std::endl;
  }
}

void allocate_and_deallocate_pool(const std::string& resource)
{
  constexpr std::size_t SIZE = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator(resource);

  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
      resource + "_pool", allocator);

  auto test_allocator = rm.makeAllocator<umpire::strategy::QuickPool>(
      resource + "_test", pooled_allocator);

  double* data =
      static_cast<double*>(pooled_allocator.allocate(SIZE * sizeof(double)));

  double* test =
      static_cast<double*>(test_allocator.allocate(SIZE * sizeof(double)));
  
  std::cout << "Allocated " << (SIZE * sizeof(double)) << " bytes using the "
            << pooled_allocator.getName() << " allocator" << std::endl;

  check_parent(pooled_allocator, resource);

  pooled_allocator.deallocate(data);

  std::cout << "Memory deallocated." << std::endl << std::endl;
  
  std::cout << "...NOW: Allocated " << (SIZE * sizeof(double)) << " bytes using the "
            << test_allocator.getName() << " allocator" << std::endl;

  check_parent(test_allocator, resource);

  test_allocator.deallocate(test);

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
