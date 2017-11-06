#include <iostream>

#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/Pool.hpp"

#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/GenericAllocationStrategyFactory.hpp"
#include "umpire/strategy/AllocationStrategyRegistry.hpp"

int main(int argc, char* argv[])
{
  auto& rm = umpire::ResourceManager::getInstance();

  std::cout << "Available allocators: ";
  for (auto s : rm.getAvailableAllocators()){
    std::cout << s << "  ";
  }
  std::cout << std::endl;

  /*
   * Register allocators.
   */
  auto& alloc_registry = umpire::strategy::AllocationStrategyRegistry::getInstance();

  alloc_registry.registerAllocationStrategy(
      std::make_shared<umpire::strategy::GenericAllocationStrategyFactory<umpire::strategy::Pool> >("POOL"));

  alloc_registry.registerAllocationStrategy(
      std::make_shared<
        umpire::strategy::GenericAllocationStrategyFactory<
          umpire::strategy::MonotonicAllocationStrategy> >("MONOTONIC"));

  /*
   * Create some allocators.
   */
  auto alloc = rm.makeAllocator("POOL", "POOL", {0,0,64}, {rm.getAllocator("HOST")});
  alloc = rm.makeAllocator("MONOTONIC 1024", "MONOTONIC", {1024,0,0}, {rm.getAllocator("HOST")});
  alloc = rm.makeAllocator("MONOTONIC 4096", "MONOTONIC", {4096,0,0}, {rm.getAllocator("HOST")});



  /*
   * Do some test allocations..
   */
  alloc = rm.getAllocator("POOL");
  void* test = alloc.allocate(100);
  alloc.deallocate(test);


  alloc = rm.getAllocator("HOST");
  test = alloc.allocate(100);
  alloc.deallocate(test);

  alloc = rm.getAllocator("MONOTONIC 1024");
  test = alloc.allocate(14);
  alloc.deallocate(test);

  std::cout << "Size: " << alloc.getSize(test) << std::endl;

  std::cout << "Available allocators: ";
  for (auto s : rm.getAvailableAllocators()){
    std::cout << s << ", ";
  }
  std::cout << std::endl;

  return 0;
}
