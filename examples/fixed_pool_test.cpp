#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/FixedPool.hpp"
#include "umpire/strategy/SizeLimiter.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  void* test[8];

  //This allocator acts as the upper limit for the total amount of 
  //memory that I want to allocate in this example.
  auto limiter_alloc = rm.makeAllocator<umpire::strategy::SizeLimiter>(
      "size_limiter", rm.getAllocator("HOST"), 1024 * 8);

  //This allocator will only be able to allocate 8 fixed-sized pools
  //before it will hit the limit imposed by the SizeLimiter
  auto fixed_alloc = rm.makeAllocator<umpire::strategy::FixedPool>(
        "fixed_pool", rm.getAllocator("size_limiter"), 1024, 1);

  //Fill the FixedPool allocator to the max.
  std::cout<<"Allocating FixedPool..."<<std::endl;
  for (int i = 0; i < 8; i++) {
    test[i] = fixed_alloc.allocate(1024);
  }

  //Deallocate all the memory given so far.  
  std::cout<<"Deallocating FixedPool..."<<std::endl;
  for (int i = 0; i < 8; i++) {
    fixed_alloc.deallocate(test[i]);
  }

  //Release the FixedPool memory space.
  std::cout<<"Releasing FixedPool..."<<std::endl;
  fixed_alloc.release();
 
  //Allocate 1024 from size limiter. If it works, then Release
  //is working ok. If not, then the memory was not properly 
  //freed by FixedPool.
  std::cout<<"Trying to allocate and deallocate with SizeLimiter..."<<std::endl;
  test[0] = limiter_alloc.allocate(1024);
  limiter_alloc.deallocate(test[0]);
  std::cout<<"Successfully reused memory with SizeLimiter!"<<std::endl;

  return 0;
}
