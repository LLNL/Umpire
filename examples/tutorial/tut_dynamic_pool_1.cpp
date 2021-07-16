//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/umpire.hpp"

template<typename Resource>
void allocate_and_deallocate_pool_t()
{
  using pool_t = umpire::strategy::slot_pool<Resource>;
  using allocator_t = umpire::allocator<double, pool_t>;

  constexpr std::size_t SIZE = 1024;

  allocator_t pool = umpire::make_allocator<double, pool_t>(
    Resource::get()->get_name() + "_typed_pool", Resource::get(), 8); 

  double* data = pool.allocate(SIZE);

  std::cout << "Allocated " << SIZE*sizeof(typename decltype(pool)::value_type)
   << " using the " << pool.get_name() << " allocator...";

  pool.deallocate(data);

  std::cout << " deallocated." << std::endl;
}

void allocate_and_deallocate_pool(const std::string& resource)
{
  constexpr std::size_t SIZE = 1024;

  umpire::allocator<double> pool = 
    umpire::make_allocator<double, umpire::strategy::slot_pool<>>(
    resource + "_pool", umpire::get_strategy(resource), 8);

  double* data = pool.allocate(SIZE);

  std::cout << "Allocated " << SIZE*sizeof(typename decltype(pool)::value_type)
   << " using the " << pool.get_name() << " allocator...";

  pool.deallocate(data);

  std::cout << " deallocated." << std::endl;
}

int main(int, char**) {
  umpire::initialize();
  allocate_and_deallocate_pool("HOST");
  allocate_and_deallocate_pool_t<umpire::resource::host_memory<>>();

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
