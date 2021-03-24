//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include <iostream>

const unsigned long my_max = 33285996544ULL;

int main() {
  auto& rm = umpire::ResourceManager::getInstance();
  const long num_alloc {8126464};
  const int sizes {4096};
  
  int resources = 4;
  umpire::Allocator allocs[resources];
  allocs[0] = rm.getAllocator("DEVICE");
  allocs[1] = rm.getAllocator("DEVICE");
  allocs[2] = rm.getAllocator("UM");
  allocs[3] = rm.getAllocator("PINNED");

  long total{0};
  int count{0};
  for(int i = 0; i < resources; i++) {
    std::cout << "Got allocator: " << allocs[i].getName() << std::endl;
    void* allocations[num_alloc];
    /* Allocate in SMALL chunks until malloc fails */
    while((total <= my_max))
    {
      try {
      allocations[count] = allocs[i].allocate(sizes);
      } catch(umpire::util::Exception&) {
        std::cout<<"Iteration: "<<count<<std::endl;
        std::cout<<"Total: "<<total<<" and that is "<<my_max-total<<"short of total available gpu memory"<<std::endl;
      }
      total+=sizes;

      if(total % 1073741824 == 0) {
        rm.memset(allocations[count], 0, sizes);
        std::cout<<"Size allocation: "<<sizes<<std::endl;
        std::cout<<"Current: "<<allocs[i].getCurrentSize()<<std::endl;
      }
      count++; 
    }

    for(int h = 0; h < count; h++)
      allocs[i].deallocate(allocations[h]);

    total = 0;
    count = 0;
  }

  std::cout<<"Done!"<<std::endl;

  return 0;
}
