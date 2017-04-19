#include "umpire/ResourceManager.hpp"

int main() {
  auto spaces = umpire::ResourceManager::getInstance()->getAvailableSpaces();

  for (auto const& space : spaces) {
    std::cout << "Found space " << space.getName() << std::endl;
  }
}
