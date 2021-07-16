#pragma once

#include "umpire/memory.hpp"

namespace umpire {
namespace strategy {
  
struct allocation_strategy :
  public memory
{
  allocation_strategy(const std::string& name) :
    memory(name) {}
};

}
}
