#include <iostream>
#include <chrono>

#include <random>

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#define ALLOCATIONS 100000

void benchmark_allocator(std::string name) {
  std::mt19937 gen(12345678);
  std::uniform_int_distribution<size_t> dist(64, 4096);

  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator(name);

  void* allocations[ALLOCATIONS];

  auto begin_alloc = std::chrono::system_clock::now();

  for (int i = 0; i < ALLOCATIONS; i++) {
    std::size_t size = dist(gen);
    allocations[i] = alloc.allocate(size);
  }

  auto end_alloc = std::chrono::system_clock::now();

  auto begin_dealloc = std::chrono::system_clock::now();
  for (int i = 0; i < ALLOCATIONS; i++) {
    alloc.deallocate(allocations[i]);
  }
  auto end_dealloc = std::chrono::system_clock::now();

  std::cout << name << std::endl;
  std::cout << "    alloc: " <<  std::chrono::duration<double>(end_alloc - begin_alloc).count()/ALLOCATIONS << std::endl;
  std::cout << "    dealloc: " << std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/ALLOCATIONS << std::endl;
}

int main(int, char**) {
  benchmark_allocator("HOST");

#if defined(UMPIRE_ENABLE_CUDA)
  benchmark_allocator("DEVICE");
  benchmark_allocator("UM");
#endif
}
