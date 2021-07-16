//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/umpire.hpp"

template<typename Resource>
void allocate_and_deallocate_t()
{
  constexpr std::size_t SIZE = 1024;

  Resource* strategy = Resource::get();
  umpire::allocator<double, Resource> allocator{strategy};

  double* data = allocator.allocate(SIZE*sizeof(double));

  std::cout << "Allocated " << SIZE*sizeof(typename decltype(allocator)::value_type)
   << " using the " << allocator.get_name() << " allocator...";

  allocator.deallocate(data);
  std::cout << " deallocated." << std::endl;
}

void allocate_and_deallocate(const std::string& resource)
{
  constexpr std::size_t SIZE = 1024;

  auto strategy = umpire::get_strategy(resource);
  umpire::allocator<double>  allocator{strategy};

  double* data = allocator.allocate(SIZE*sizeof(double));

  std::cout << "Allocated " << SIZE*sizeof(typename decltype(allocator)::value_type)
   << " using the " << allocator.get_name() << " allocator...";

  allocator.deallocate(data);
  std::cout << " deallocated." << std::endl;
}

int main(int, char**) {
  umpire::initialize();

  allocate_and_deallocate("HOST");
  allocate_and_deallocate_t<umpire::resource::host_memory<>>();

#if defined(UMPIRE_ENABLE_CUDA)
  allocate_and_deallocate("DEVICE");
  allocate_and_deallocate("UM");
  allocate_and_deallocate("PINNED");
#endif
#if defined(UMPIRE_ENABLE_HIP)
  allocate_and_deallocate("DEVICE");
  allocate_and_deallocate("PINNED");
#endif

  return 0;
}
