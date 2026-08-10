#include <iostream>
#include <chrono>
#include <random>
#include <typeinfo>

#include "allocators.hpp"

#define ALLOCATIONS 100000

template <typename Allocator>
void benchmark_allocator(Allocator alloc, bool TimeIntrospection=false) {
  std::mt19937 gen(12345678);
  std::uniform_int_distribution<std::size_t> dist(64, 4096);
  int errors{0};

  void* allocations[ALLOCATIONS];
  std::size_t sizes[ALLOCATIONS];
  auto begin_alloc = std::chrono::system_clock::now();

  for (int i = 0; i < ALLOCATIONS; i++) {
    std::size_t size = dist(gen);
    allocations[i] = alloc.allocate(size);
    sizes[i] = size;
  }

  auto end_alloc = std::chrono::system_clock::now();

  auto begin_introspection = std::chrono::system_clock::now();
  if (TimeIntrospection) {

    for (int i = 0; i < ALLOCATIONS; i++) {
      auto record = alloc.getRecord(allocations[i]);
      if (sizes[i] != record->size) {
        errors++;
      }
    }

  }
  auto end_introspection = std::chrono::system_clock::now();

  auto begin_dealloc = std::chrono::system_clock::now();
  for (int i = 0; i < ALLOCATIONS; i++) {
    alloc.deallocate(allocations[i]);
  }
  auto end_dealloc = std::chrono::system_clock::now();

  std::cout << typeid(alloc).name() << std::endl;
  std::cout << "    alloc: " <<  std::chrono::duration<double>(end_alloc - begin_alloc).count()/ALLOCATIONS << std::endl;
  if (TimeIntrospection) {
  std::cout << "    ??: " <<  std::chrono::duration<double>(end_introspection - begin_introspection).count()/ALLOCATIONS << std::endl;
  std::cout << "    errors: " << errors << std::endl;
  }
  std::cout << "    dealloc: " << std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/ALLOCATIONS << std::endl;
}

int main(int, char**) {
  benchmark_allocator(umpire::mallocator{});
  benchmark_allocator(umpire::header_allocator<umpire::host_allocator_tag>{}, true);


  benchmark_allocator(umpire::cudamallocator{});
  benchmark_allocator(umpire::header_allocator<umpire::cuda_allocator_tag>{}, true);

  benchmark_allocator(umpire::cudaummallocator{});
  benchmark_allocator(umpire::header_allocator<umpire::cuda_um_allocator_tag>{}, true);
}
