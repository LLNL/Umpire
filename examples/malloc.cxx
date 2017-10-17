#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#include <iostream>

int main() {
  auto &rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");

  const int size = 100;

  double* my_array = static_cast<double*>(alloc.allocate(100 * sizeof(double)));

  for (int i = 0; i < size; i++) {
    my_array[i] = static_cast<double>(i);
  }

  for (int i = 0; i < size; i++) {
    std::cout << my_array[i] << " should be " << i << std::endl;
  }

  alloc.deallocate(my_array);
}
