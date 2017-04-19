#include "umpire/Umpire.hpp"

#include <iostream>

int main() {

  const int size = 100;

  double* my_array = static_cast<double*>(umpire::malloc(100 * sizeof(double)));

  for (int i = 0; i < size; i++) {
    my_array[i] = static_cast<double>(i);
  }

  for (int i = 0; i < size; i++) {
    std::cout << my_array[i] << " should be " << i << std::endl;
  }

  umpire::free(my_array);
}
