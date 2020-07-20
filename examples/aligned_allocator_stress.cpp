#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/AlignedAllocator.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"

#include <iostream>
#include <chrono>
#include <random>
#include <string>
#include <vector>

int main()
{
  auto& rm(umpire::ResourceManager::getInstance());
  auto myAlloc{ rm.getAllocator("HOST") };

  const std::size_t alignment=64;
  auto alignedAlloc{ rm.makeAllocator<umpire::strategy::AlignedAllocator>("HOST_aligned", myAlloc, alignment) };

  const std::size_t block_size{ 12ull * 1024ull * 1024ull * 1024ull };
  auto pool{ rm.makeAllocator<umpire::strategy::DynamicPool>("HOST_pool", alignedAlloc, block_size, block_size, alignment) };
  auto alloc{ rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>("HOST_safe_pool", pool) };
  auto dynamic_pool = umpire::util::unwrap_allocator<umpire::strategy::DynamicPool>(pool);

  if (! dynamic_pool) {
    UMPIRE_ERROR( pool.getName() << " is not a DynamicPool, cannot get statistics" );
  }

  const std::size_t min_alloc_size{1};
  const std::size_t max_alloc_size{ 16ull * 1024ull * 1024ull };
  const std::size_t max_total_size{ 1024ull * 1024ull *1024ull };
  const int max_allocations{10000};

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<std::size_t> distrib(min_alloc_size, max_alloc_size);
  int exceptions_thrown{0};

  for (int j{0}; j < 100000000; ++j) {
    std::vector<void*> allocations;
    std::size_t total_bytes{0};

    try {
      for (int i{0}; i < max_allocations; ++i) {
        std::size_t nbytes{distrib(gen)};
        void *ptr{alloc.allocate(nbytes)};

        if (0 != (reinterpret_cast<std::ptrdiff_t>(ptr)%alignment)) {
          std::cerr << "Request for aligned pointer from "
            << "HOST_safe_pool"
            << " resulted in unaligned pointer. (" << ptr << ")" << std::endl;
            return -1;
        }

        allocations.push_back(alloc.allocate(nbytes));
        total_bytes += nbytes;
        if (total_bytes >= max_total_size) {
          break;
        }
      }
    } catch (...) {
      exceptions_thrown++;
      std::cout << "Thrown Exception Caught: " << exceptions_thrown << std::endl;
    }

    std::cout << "Iteration: " << j
      << " - " << total_bytes << " allocated across "
      << allocations.size() << " allocations - "
      << " Current Size: " << dynamic_pool->getCurrentSize()
      << " Actual Size: " << dynamic_pool->getActualSize()
      << " High Watermark: " << dynamic_pool->getHighWatermark();

    for ( auto a : allocations ) {
      alloc.deallocate(a);
    }

    dynamic_pool->coalesce();
    dynamic_pool->release();

    std::cout
      << " Current Size: " << dynamic_pool->getCurrentSize()
      << " Actual Size: " << dynamic_pool->getActualSize()
      << " High Watermark: " << dynamic_pool->getHighWatermark()
      << std::endl;

  }
  std::cout << "Goodbye" << std::endl;
  return 0;

}