#include <stdio.h>
#include <math.h>
#include <iostream>

#include <thread>
#include <chrono>

#include "camp/camp.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/ResourceAwarePool.hpp"
#include "umpire/strategy/QuickPool.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
using resource_type = camp::resources::Cuda;
#elif defined(UMPIRE_ENABLE_HIP)
using resource_type = camp::resources::Hip;
#endif

constexpr int SIZE = 1 << 18;

void test_rap()
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool", rm.getAllocator("DEVICE"));

  int ROUNDS[5] = {16, 32, 64, 128, 256};

  for(int f = 0; f < 5; f++) {
    const int NUM_ALLOC = ROUNDS[f];
    std::cout << std::endl << "Number of Allocations: " << ROUNDS[f] << std::endl;

    double* a[NUM_ALLOC];

    // Create camp resources for device streams
    resource_type d1, d2;

    //Fill the pending list
    auto start_total1a = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      a[i] = static_cast<double*>(pool.allocate(SIZE * sizeof(double), d1));
    }
    auto end_total1a = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total1a = end_total1a - start_total1a;
    std::cout << "Execution time for allocating: " << (duration_total1a.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;

    auto start_total1 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      pool.deallocate(a[i], d1);
    }
    auto end_total1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total1 = end_total1 - start_total1;
    std::cout << "Execution time for deallocating: " << (duration_total1.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;

    //Reallocate with the same resource
    {
      auto start_total = std::chrono::high_resolution_clock::now();
      for( int i = 0; i < NUM_ALLOC; i ++) {
        a[i] = static_cast<double*>(pool.allocate(SIZE * sizeof(double), d1));
      }
      auto end_total = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_total1 = end_total - start_total;
      std::cout << "Total execution time for reallocating with the SAME resource: " << (duration_total1.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
    }

    auto start_total2 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      pool.deallocate(a[i], d1);
    }
    auto end_total2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total2 = end_total2 - start_total2;
    std::cout << "Execution time for deallocating: " << (duration_total2.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
  
    //Fill the pending list
    auto start_total2a = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      a[i] = static_cast<double*>(pool.allocate(SIZE * sizeof(double), d1));
    }
    auto end_total2a = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total2a = end_total2a - start_total2a;
    std::cout << "Execution time for allocating: " << (duration_total2a.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;

    auto start_total3 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      pool.deallocate(a[i], d1);
    }
    auto end_total3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total3 = end_total3 - start_total3;
    std::cout << "Execution time for deallocating: " << (duration_total3.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;

    //Reallocate with different resource
    {
      auto start_total = std::chrono::high_resolution_clock::now();
      for( int i = 0; i < NUM_ALLOC; i++) {
        a[i] = static_cast<double*>(pool.allocate(SIZE * sizeof(double), d2));
      }
      auto end_total = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_total1 = end_total - start_total;
      std::cout << "Total execution time for reallocating with a DIFFERENT resource: " << (duration_total1.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
    }

    auto start_total4 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      pool.deallocate(a[i], d2);
    }
    auto end_total4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total4 = end_total4 - start_total4;
    std::cout << "Execution time for deallocating: " << (duration_total4.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
  }
}

void test_qp()
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("qp-pool", rm.getAllocator("DEVICE"));

  int ROUNDS[5] = {16, 32, 64, 128, 256};

  for(int f = 0; f < 5; f++) {
    const int NUM_ALLOC = ROUNDS[f];
    std::cout << std::endl << "Number of Allocations: " << ROUNDS[f] << std::endl;

    double* a[NUM_ALLOC];

    //Fill the pending list
    auto start_total1a = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      a[i] = static_cast<double*>(pool.allocate(SIZE * sizeof(double)));
    }
    auto end_total1a = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total1a = end_total1a - start_total1a;
    std::cout << "Execution time for allocating: " << (duration_total1a.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;

    auto start_total1 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      pool.deallocate(a[i]);
    }
    auto end_total1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total1 = end_total1 - start_total1;
    std::cout << "Execution time for deallocating1: " << (duration_total1.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;

    //Reallocate with the same resource
    {
      auto start_total = std::chrono::high_resolution_clock::now();
      for( int i = 0; i < NUM_ALLOC; i ++) {
        a[i] = static_cast<double*>(pool.allocate(SIZE * sizeof(double)));
      }
      auto end_total = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_total1 = end_total - start_total;
      std::cout << "Total execution time for reallocating with the SAME resource: " << (duration_total1.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
    }

    auto start_total2 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      pool.deallocate(a[i]);
    }
    auto end_total2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total2 = end_total2 - start_total2;
    std::cout << "Execution time for deallocating2: " << (duration_total2.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
  
    //Fill the pending list
    auto start_total2a = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      a[i] = static_cast<double*>(pool.allocate(SIZE * sizeof(double)));
    }
    auto end_total2a = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total2a = end_total2a - start_total2a;
    std::cout << "Execution time for allocating: " << (duration_total2a.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;

    auto start_total3 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      pool.deallocate(a[i]);
    }
    auto end_total3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total3 = end_total3 - start_total3;
    std::cout << "Execution time for deallocating3: " << (duration_total3.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;

    //Reallocate with different resource
    {
      auto start_total = std::chrono::high_resolution_clock::now();
      for( int i = 0; i < NUM_ALLOC; i++) {
        a[i] = static_cast<double*>(pool.allocate(SIZE * sizeof(double)));
      }
      auto end_total = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_total1 = end_total - start_total;
      std::cout << "Total execution time for reallocating with a DIFFERENT resource: " << (duration_total1.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
    }

    auto start_total4 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      pool.deallocate(a[i]);
    }
    auto end_total4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total4 = end_total4 - start_total4;
    std::cout << "Execution time for deallocating4: " << (duration_total4.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
  }
}

void test_device_alloc()
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto not_pool = rm.getAllocator("DEVICE");

  int ROUNDS[5] = {16, 32, 64, 128, 256};

  for(int f = 0; f < 5; f++) {
    const int NUM_ALLOC = ROUNDS[f];
    std::cout << std::endl << "Number of Allocations: " << ROUNDS[f] << std::endl;

    double* a[NUM_ALLOC];

    //Fill the pending list
    auto start_total1a = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      a[i] = static_cast<double*>(not_pool.allocate(SIZE * sizeof(double)));
    }
    auto end_total1a = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total1a = end_total1a - start_total1a;
    std::cout << "Execution time for allocating: " << (duration_total1a.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;

    auto start_total1 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      not_pool.deallocate(a[i]);
    }
    auto end_total1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total1 = end_total1 - start_total1;
    std::cout << "Execution time for deallocating1: " << (duration_total1.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;

    //Reallocate with the same resource
    {
      auto start_total = std::chrono::high_resolution_clock::now();
      for( int i = 0; i < NUM_ALLOC; i ++) {
        a[i] = static_cast<double*>(not_pool.allocate(SIZE * sizeof(double)));
      }
      auto end_total = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_total1 = end_total - start_total;
      std::cout << "Total execution time for reallocating with the SAME resource: " << (duration_total1.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
    }

    auto start_total2 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      not_pool.deallocate(a[i]);
    }
    auto end_total2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total2 = end_total2 - start_total2;
    std::cout << "Execution time for deallocating2: " << (duration_total2.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
  
    //Fill the pending list
    auto start_total2a = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      a[i] = static_cast<double*>(not_pool.allocate(SIZE * sizeof(double)));
    }
    auto end_total2a = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total2a = end_total2a - start_total2a;
    std::cout << "Execution time for allocating: " << (duration_total2a.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;

    auto start_total3 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      not_pool.deallocate(a[i]);
    }
    auto end_total3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total3 = end_total3 - start_total3;
    std::cout << "Execution time for deallocating3: " << (duration_total3.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;

    //Reallocate with different resource
    {
      auto start_total = std::chrono::high_resolution_clock::now();
      for( int i = 0; i < NUM_ALLOC; i++) {
        a[i] = static_cast<double*>(not_pool.allocate(SIZE * sizeof(double)));
      }
      auto end_total = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_total1 = end_total - start_total;
      std::cout << "Total execution time for reallocating with a DIFFERENT resource: " << (duration_total1.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
    }

    auto start_total4 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      not_pool.deallocate(a[i]);
    }
    auto end_total4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total4 = end_total4 - start_total4;
    std::cout << "Execution time for deallocating4: " << (duration_total4.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
  }
}

int main(int, char**)
{
  std::cout << "--------Starting ResourceAwarePool tests--------" << std::endl;
  test_rap();
  std::cout << "----------------" << std::endl;
  std::cout << "--------Starting QuickPool tests--------" << std::endl;
  test_qp();
  std::cout << "----------------" << std::endl;
  std::cout << "--------Starting DEVICE alloc tests--------" << std::endl;
  test_device_alloc();
  std::cout << "----------------" << std::endl;

  return 0;
}
  
