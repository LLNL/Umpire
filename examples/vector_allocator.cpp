#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#include "umpire/TypedAllocator.hpp"

int main(int argc, char* argv[]) {
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("HOST");

  umpire::TypedAllocator<double> vector_allocator(alloc);

  std::vector< double, umpire::TypedAllocator<double> > my_vector(vector_allocator);

  my_vector.resize(100);

  my_vector[50] = 3.14;

  return 0;
}
