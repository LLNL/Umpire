//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/util/wrap_allocator.hpp"

void print_stats(umpire::strategy::QuickPool* alloc)
{
  std::cout << "Actual size is :" << alloc->getActualSize() << std::endl;
  std::cout << "Current size is :" << alloc->getCurrentSize() << std::endl;
  std::cout << "Releaseable size is :" << alloc->getReleasableSize() << std::endl;
  std::cout << "--------------------------------------" << std::endl;
}

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  auto heuristic_function = umpire::strategy::QuickPool::percent_releasable(75);
  //auto heuristic_function = umpire::strategy::QuickPool::blocks_releasable(2);

  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("quick_pool", allocator, 1024ul, 1024ul, 16, heuristic_function);
  auto stats_alloc = umpire::util::unwrap_allocator<umpire::strategy::QuickPool>(pool);

  constexpr std::size_t SIZE{1 << 30};

  double* data = static_cast<double*>(pool.allocate(SIZE * sizeof(double)));

  for (size_t i = 0; i < SIZE; i++) {
    data[i] = 2 * i - i;
  }
  std::cout << "After first alloc:" << std::endl;
  print_stats(stats_alloc);

  double* data2 = static_cast<double*>(pool.allocate(2 * SIZE * sizeof(double)));

  for (size_t i = 10; i < SIZE; i+=5) {
    data2[i] = 2 * i - i;
  }
  
  //Deallocate first data
  pool.deallocate(data);

  std::cout << "After second alloc and first dealloc:" << std::endl;
  print_stats(stats_alloc);

  double* data3 = static_cast<double*>(pool.allocate(3 * SIZE * sizeof(double)));

  for (size_t i = 5; i < SIZE; i+=5) {
    data3[i] = 2 * i - i;
  }

  // Deallocate second data
  pool.deallocate(data2);

  std::cout << "After third alloc and second dealloc:" << std::endl;
  print_stats(stats_alloc);

  double* data4 = static_cast<double*>(pool.allocate(SIZE * sizeof(double)));

  for (size_t i = 0; i < SIZE; i++) {
    data4[i] = 2 * i - i;
  }
  std::cout << "After fourth alloc:" << std::endl;
  print_stats(stats_alloc);

  double* data5 = static_cast<double*>(pool.allocate(2* SIZE * sizeof(double)));

  for (size_t i = 10; i < 2 * SIZE; i+=5) {
    data5[i] = 2 * i - i;
  }

  // Deallocate forth data
  pool.deallocate(data4);

  std::cout << "After fifth alloc and fouth dealloc:" << std::endl;
  print_stats(stats_alloc);

  stats_alloc->coalesce();
  data = static_cast<double*>(pool.allocate(SIZE * sizeof(double)));

  for (size_t i = 0; i < SIZE; i++) {
    data[i] = 2 * i - i;
  }
  std::cout << "After redo of first alloc:" << std::endl;
  print_stats(stats_alloc);

  // Deallocate third data
  pool.deallocate(data3);
  pool.deallocate(data5);

  std::cout << "After deallocs, but with leaks:" << std::endl;
  print_stats(stats_alloc);

  return 0;
}
