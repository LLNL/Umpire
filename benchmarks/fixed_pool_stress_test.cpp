#include <chrono>
#include <string>
#include <random>
#include <numeric>

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#define CONVERT 1000000 //convert sec (s) to microsec (us)

const int NUM_RND {1000}; //number of rounds (used to average timing)
const int NUM_ALLOC {512}; //number of allocations used for testing
const int OBJECTS_PER_BLOCK {1<<11}; //number of blocks of object_bytes size (2048)

void run_test(umpire::Allocator alloc, int SIZE, std::vector<int> indices, std::string test_name)
{
  double time[2] = {0.0, 0.0};
  void* allocations[NUM_ALLOC];

  for(int i = 0; i < NUM_RND; i++) {
    auto begin_alloc{std::chrono::system_clock::now()};
    for (int j = 0; j < NUM_ALLOC; j++)
      allocations[indices[j]] = alloc.allocate(SIZE);
    auto end_alloc{std::chrono::system_clock::now()};
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/NUM_ALLOC;

    auto begin_dealloc {std::chrono::system_clock::now()};
    for (int h = 0; h < NUM_ALLOC; h++)
      alloc.deallocate(allocations[indices[h]]);
    auto end_dealloc {std::chrono::system_clock::now()};
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/NUM_ALLOC;
  }

  alloc.release();
  double alloc_t{(time[0]/double(NUM_RND)*CONVERT)};
  double dealloc_t{(time[1]/double(NUM_RND)*CONVERT)};

  std::cout << "  " << test_name << " (FixedPool - " << (long long int)SIZE*OBJECTS_PER_BLOCK << "):" << std::endl; 
  std::cout << "    alloc: " << alloc_t << "(us)" << std::endl;
  std::cout << "    dealloc: " << dealloc_t << "(us)" << std::endl;
  std::cout << "    lifetime: " << alloc_t+dealloc_t << "(us)" << std::endl << std::endl;
}

int main(int, char**)
{
  //Set up formatting for output
  std::cout << std::fixed << std::setprecision(9); 

  auto& rm{umpire::ResourceManager::getInstance()};
  umpire::Allocator alloc{rm.getAllocator("HOST")};
  
  //Using two different sizes to show average times don't vary much.
  const int NUM_SIZES {2}; 

  //Array of sizes used are 67108864 vs. 2048 (large vs. small)
  static int SIZE[NUM_SIZES] {1<<26, 1<<11};

  //create vector of indices for "same_order" tests
  std::vector<int> same_order_index(NUM_ALLOC);
  for(int i = 0; i < NUM_ALLOC; i++) 
    same_order_index[i] = i;

  //create vector of indices for "reverse_order" tests
  std::vector<int> reverse_order_index(same_order_index);
  std::reverse(reverse_order_index.begin(), reverse_order_index.end());

  //create vector of indices for "shuffle_order" tests
  std::vector<int> shuffle_order_index(same_order_index);
  std::mt19937 gen(NUM_ALLOC);
  std::shuffle(&shuffle_order_index[0], &shuffle_order_index[NUM_ALLOC], gen);
 
  //create the FixedPool allocator and run stress tests for all sizes
  for(int i = 0; i < NUM_SIZES; i++)
  {
    umpire::Allocator pool_alloc = rm.makeAllocator<umpire::strategy::FixedPool, false>
                                 ("fixed_pool" + std::to_string(i), alloc, SIZE[i], OBJECTS_PER_BLOCK);
    run_test(pool_alloc, SIZE[i], same_order_index, "SAME_ORDER");
    run_test(pool_alloc, SIZE[i], reverse_order_index, "REVERSE_ORDER");
    run_test(pool_alloc, SIZE[i], shuffle_order_index, "SHUFFLE_ORDER");
  }

  return 0;
}
