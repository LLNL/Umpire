#include "umpire/ResourceManager.hpp"

int main() {
  auto spaces = umpire::ResourceManager::getInstance()->getAvailableSpaces();

  for (auto const& space : spaces) {
    std::cout << "Found resource " << space.getName() << std::endl;
  }
}
