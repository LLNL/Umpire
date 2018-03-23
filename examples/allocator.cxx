#include "umpire/Allocator.hpp"

int main(int argc, char* argv[]) {
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("HOST");

  std::cout << "Available allocators: ";
  for (auto s : rm.getAvailableAllocators()){
    std::cout << s << "  ";
  }
  std::cout << std::endl;

  return 0;
}
