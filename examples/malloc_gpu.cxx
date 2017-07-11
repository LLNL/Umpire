#include "umpire/Umpire.hpp"
#include "umpire/ResourceManager.hpp"

#include <iostream>

int main() {
  const int size = 100;

  umpire::ResourceManager rm = umpire::ResourceManager::getInstance();
  auto space = rm.getSpace("DEVICE");
  double* my_array = static_cast<double*>(rm.allocate(size * sizeof(double), space));

  umpire::free(my_array);
}
