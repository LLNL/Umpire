#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/Pool.hpp"

int main(int argc, char* argv[])
{
  auto& rm = umpire::ResourceManager::getInstance();

  rm.registerAllocator<umpire::strategy::Pool>("POOL_ONE", "HOST");
  rm.registerAllocator<umpire::strategy::Pool>("POOL_TWO", "HOST");

  auto pool = rm.getAllocator("POOL_ONE");
  void* test = pool.allocate(100);
  pool.deallocate(test);

  pool = rm.getAllocator("POOL_TWO");
  test = pool.allocate(100);
  pool.deallocate(test);

  return 0;
}
