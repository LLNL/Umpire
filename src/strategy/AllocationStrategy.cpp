#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

AllocationStrategy::AllocationStrategy(const std::string& name, int id) :
  m_name(name),
  m_id(id)
{
}

std::string
AllocationStrategy::getName()
{
  return m_name;
}

int
AllocationStrategy::getId()
{
  return m_id;
}

} // end of namespace strategy
} // end of namespace umpire
