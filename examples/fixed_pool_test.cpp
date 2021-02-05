#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/FixedPool.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto m_alloc = rm.makeAllocator<umpire::strategy::FixedPool>(
        "fixed_pool1", rm.getAllocator("HOST"), 1<<28);

  for (int i = 0; i < 100000; i++) {
    void* test = m_alloc.allocate(1<<28);
    m_alloc.deallocate(test);
    m_alloc.release();
  }
 
  return 0;
}
