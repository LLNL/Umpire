#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <random>

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/strategy/FixedPool.hpp"

#define CONVERT 1000000 //convert sec (s) to microsec (us)
const int NUM_RND = 1000; //number of rounds (used to average timing)
const int NUM_ALLOC = 512;
const int NUM_SIZES = 5;

void same_order(umpire::Allocator alloc, int SIZE)
{
  double time[2] = {0.0, 0.0};
  void* allocations[NUM_ALLOC];

  for(int i = 0; i < NUM_RND; i++) {
    auto begin_alloc = std::chrono::system_clock::now();
    for (int j = 0; j < NUM_ALLOC; j++)
      allocations[j] = alloc.allocate(SIZE);
    auto end_alloc = std::chrono::system_clock::now();
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/NUM_ALLOC;

    auto begin_dealloc = std::chrono::system_clock::now();
    for (int h = 0; h < NUM_ALLOC; h++)
      alloc.deallocate(allocations[h]);
    auto end_dealloc = std::chrono::system_clock::now();
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/NUM_ALLOC;
  }

  alloc.release();
  double alloc_t{(time[0]/double(NUM_RND)*CONVERT)};
  double dealloc_t{(time[1]/double(NUM_RND)*CONVERT)};

  std::cout << "  SAME_ORDER (FixedPool - " << (long long int)NUM_ALLOC*SIZE << "):" << std::endl; 
  std::cout << "    alloc: " << alloc_t << "(us)" << std::endl;
  std::cout << "    dealloc: " << dealloc_t << "(us)" << std::endl;
  std::cout << "    lifetime: " << alloc_t+dealloc_t << "(us)" << std::endl << std::endl;
}

void reverse_order(umpire::Allocator alloc, int SIZE)
{
  double time[2] = {0.0, 0.0};
  void* allocations[NUM_ALLOC];

  for(int i = 0; i < NUM_RND; i++) {
    auto begin_alloc = std::chrono::system_clock::now();
    for (int j = 0; j < NUM_ALLOC; j++)
      allocations[j] = alloc.allocate(SIZE);
    auto end_alloc = std::chrono::system_clock::now();
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/NUM_ALLOC;

    auto begin_dealloc = std::chrono::system_clock::now();
    for (int h = (NUM_ALLOC-1); h >=0; h--)
      alloc.deallocate(allocations[h]);
    auto end_dealloc = std::chrono::system_clock::now();
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/NUM_ALLOC;
  }

  alloc.release();
  double alloc_t{(time[0]/double(NUM_RND)*CONVERT)};
  double dealloc_t{(time[1]/double(NUM_RND)*CONVERT)};

  std::cout << "  REVERSE_ORDER (FixedPool - " << (long long int)NUM_ALLOC*SIZE << "):" << std::endl; 
  std::cout << "    alloc: " << alloc_t << "(us)" << std::endl;
  std::cout << "    dealloc: " << dealloc_t << "(us)" << std::endl;
  std::cout << "    lifetime: " << alloc_t+dealloc_t << "(us)" << std::endl << std::endl;
}

void shuffle(umpire::Allocator alloc, int SIZE)
{
  std::mt19937 gen(NUM_ALLOC);
  double time[2] = {0.0, 0.0};
  void* allocations[NUM_ALLOC];

  for(int i = 0; i < NUM_RND; i++) {
    auto begin_alloc = std::chrono::system_clock::now();
    for (int j = 0; j < NUM_ALLOC; j++)
      allocations[j] = alloc.allocate(SIZE);
    auto end_alloc = std::chrono::system_clock::now();
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/NUM_ALLOC;

    std::shuffle(&allocations[0], &allocations[NUM_ALLOC], gen);
    auto begin_dealloc = std::chrono::system_clock::now();
    for (int h = 0; h < NUM_ALLOC; h++)
      alloc.deallocate(allocations[h]);
    auto end_dealloc = std::chrono::system_clock::now();
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/NUM_ALLOC;
  }

  alloc.release();
  double alloc_t{(time[0]/double(NUM_RND)*CONVERT)};
  double dealloc_t{(time[1]/double(NUM_RND)*CONVERT)};

  std::cout << "  SHUFFLE (FixedPool - " << (long long int)NUM_ALLOC*SIZE << "):" << std::endl; 
  std::cout << "    alloc: " << alloc_t << "(us)" << std::endl;
  std::cout << "    dealloc: " << dealloc_t << "(us)" << std::endl;
  std::cout << "    lifetime: " << alloc_t+dealloc_t << "(us)" << std::endl << std::endl;
}

////////////////////////////////////////////////////////////////////
//Note: the total size of allocated memory is the multiplication of
//SIZE and the default objects_per_pool value of 2048.
////////////////////////////////////////////////////////////////////
int main(int, char**)
{
  //Set up formatting for output
  std::cout << std::fixed << std::setprecision(9); 

  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("HOST");
  
  //array of all sizes used: 2^26, 2^24, 2^20, 2^15, 2^11
  static int SIZE[NUM_SIZES] = {67108864, 16777216, 1048576, 32768, 2048};

  //create the FixedPool allocator and run stress tests for all sizes
  for(int i = 0; i < NUM_SIZES; i++)
  {
    umpire::Allocator pool_alloc = rm.makeAllocator<umpire::strategy::FixedPool, false>
                                 ("fixed_pool" + std::to_string(i), alloc, SIZE[i]);
    same_order(pool_alloc, SIZE[i]);
    reverse_order(pool_alloc, SIZE[i]);
    shuffle(pool_alloc, SIZE[i]);
  }

  return 0;
}