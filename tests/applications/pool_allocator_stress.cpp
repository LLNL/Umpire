//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/AlignedAllocator.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/DynamicPoolMap.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"

template <typename T>
void report_and_deallocate(int iteration, std::vector<void*>& ptrs,
                           umpire::Allocator& allocator, T strategy)
{
  auto current_size = allocator.getCurrentSize();
  auto actual_size = allocator.getActualSize();
  auto high_watermark = allocator.getHighWatermark();

  for (auto a : ptrs) {
    allocator.deallocate(a);
  }

  strategy->coalesce();
  strategy->release();

  std::cout << std::setw(22) << allocator.getName() << ": " << iteration
            << std::setw(12) << current_size << " allocated across "
            << std::setw(4) << ptrs.size() << " allocations."
            << " Actual Size: " << std::setw(12) << actual_size
            << " High Watermark: " << std::setw(12) << high_watermark
            << " Adjusted - Current Size: " << std::setw(12)
            << allocator.getCurrentSize()
            << " Adjusted - Actual Size: " << std::setw(12)
            << allocator.getActualSize() << std::endl;
}

void allocate_and_check(std::size_t nbytes, std::size_t alignment,
                        std::vector<void*>& ptrs, umpire::Allocator& allocator)
{
  void* ptr{allocator.allocate(nbytes)};

  if (0 != (reinterpret_cast<std::ptrdiff_t>(ptr) % alignment)) {
    std::cerr << "Request for aligned pointer from " << allocator.getName()
              << " resulted in unaligned pointer. (" << ptr << ")" << std::endl;
    exit(1);
  }

  ptrs.push_back(ptr);
}

int main()
{
  const bool run_quick{true};
  const bool run_map{true};
  const bool run_list{true};
  const std::size_t one_megabyte{1024 * 1024};
  const std::size_t one_gigabyte{one_megabyte * 1024};
  const std::size_t allocation_alignment{64};
  const std::size_t initial_pool_size{12ull * one_gigabyte};
  const std::size_t subsequent_pool_increments{1024};
  const std::size_t min_alloc_size{1};
  const std::size_t max_alloc_size{32ull * one_megabyte};
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<std::size_t> distrib(min_alloc_size,
                                                     max_alloc_size);
  const std::size_t max_total_size{one_gigabyte};

  int exceptions_thrown{0};
  const int max_allocations{5000};
  const int max_iterations{1000000};

  auto& rm = umpire::ResourceManager::getInstance();

  auto quick_allocation_pool = rm.makeAllocator<umpire::strategy::QuickPool>(
      "HOST_quick_pool", rm.getAllocator("HOST"), initial_pool_size,
      subsequent_pool_increments);
  auto quick_dynamic_pool =
      umpire::util::unwrap_allocator<umpire::strategy::QuickPool>(
          quick_allocation_pool);
  auto quick_aligned_allocator =
      rm.makeAllocator<umpire::strategy::AlignedAllocator>(
          "HOST_quick_aligned", quick_allocation_pool, allocation_alignment);
  auto quick_alloc = rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      "HOST_quick_safe_pool", quick_aligned_allocator);

  auto map_allocation_pool = rm.makeAllocator<umpire::strategy::DynamicPoolMap>(
      "HOST_map_pool", rm.getAllocator("HOST"), initial_pool_size,
      subsequent_pool_increments);
  auto map_dynamic_pool =
      umpire::util::unwrap_allocator<umpire::strategy::DynamicPoolMap>(
          map_allocation_pool);
  auto map_aligned_allocator =
      rm.makeAllocator<umpire::strategy::AlignedAllocator>(
          "HOST_map_aligned", map_allocation_pool, allocation_alignment);
  auto map_alloc = rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      "HOST_map_safe_pool", map_aligned_allocator);

  auto list_allocation_pool =
      rm.makeAllocator<umpire::strategy::DynamicPoolList>(
          "HOST_list_pool", rm.getAllocator("HOST"), initial_pool_size,
          subsequent_pool_increments);
  auto list_dynamic_pool =
      umpire::util::unwrap_allocator<umpire::strategy::DynamicPoolList>(
          list_allocation_pool);
  auto list_aligned_allocator =
      rm.makeAllocator<umpire::strategy::AlignedAllocator>(
          "HOST_list_aligned", list_allocation_pool, allocation_alignment);
  auto list_alloc = rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      "HOST_list_safe_pool", list_aligned_allocator);

  for (int j{0}; j < max_iterations; ++j) {
    std::vector<void*> quick_allocations;
    std::vector<void*> map_allocations;
    std::vector<void*> list_allocations;
    std::size_t total_bytes{0};

    try {
      for (int i{0}; i < max_allocations && total_bytes <= max_total_size;
           ++i) {
        std::size_t nbytes{distrib(gen)};

        if (run_quick) {
          allocate_and_check(nbytes, allocation_alignment, quick_allocations,
                             quick_alloc);
        }

        if (run_map) {
          allocate_and_check(nbytes, allocation_alignment, map_allocations,
                             map_alloc);
        }

        if (run_list) {
          allocate_and_check(nbytes, allocation_alignment, list_allocations,
                             list_alloc);
        }

        total_bytes += nbytes;
      }
    } catch (...) {
      exceptions_thrown++;
      std::cout << "Thrown Exception Caught: " << exceptions_thrown
                << std::endl;
    }

    if (run_quick) {
      report_and_deallocate(j, quick_allocations, quick_alloc,
                            quick_dynamic_pool);
    }

    if (run_map) {
      report_and_deallocate(j, map_allocations, map_alloc, map_dynamic_pool);
    }

    if (run_list) {
      report_and_deallocate(j, list_allocations, list_alloc, list_dynamic_pool);
    }
    std::cout << std::endl;
  }
  std::cout << "Goodbye" << std::endl;
  return 0;
}
