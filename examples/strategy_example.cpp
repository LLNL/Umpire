#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/Pool.hpp"

int main(int argc, char* argv[])
{

  auto& rm = umpire::ResourceManager::getInstance();

  auto pool = rm.getAllocator("POOL");

  void* test = pool.allocate(100);

  pool.deallocate(test);

  return 0;
}
