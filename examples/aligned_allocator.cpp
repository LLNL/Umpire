#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/AlignedAllocator.hpp"
#include "umpire/Allocator.hpp"

#include <iostream>

int main() {
  auto& rm = umpire::ResourceManager::getInstance();
  auto aligned_alloc = rm.makeAllocator<umpire::strategy::AlignedAllocator>(
    "aligned_allocator", rm.getAllocator("HOST"), 256);

  void* data = aligned_alloc.allocate(1234);
  aligned_alloc.deallocate(data);

  data = aligned_alloc.allocate(7);
  aligned_alloc.deallocate(data);

  data = aligned_alloc.allocate(5555);
  aligned_alloc.deallocate(data);

  return 0;
}
