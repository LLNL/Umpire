#include "umpire/Allocator.hpp"

int main(int argc, char* argv[]) {
  umpire::ResourceManager rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getSpace("HOST");

  return 0;
}
