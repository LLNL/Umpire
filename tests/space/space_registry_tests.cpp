#include "gtest/gtest.h"

#include "umpire/space/MemorySpaceRegistry.hpp"
#include "umpire/space/HostSpaceFactory.hpp"

#include "umpire/util/Exception.hpp"

TEST(MemorySpaceRegistry, FindHostSpace)
{
  umpire::space::MemorySpaceRegistry& registry =
    umpire::space::MemorySpaceRegistry::getInstance();

  std::shared_ptr<umpire::space::MemorySpaceFactory> space = 
    registry.getMemorySpaceFactory("HOST");

  ASSERT_NE(nullptr, dynamic_cast<umpire::space::HostSpaceFactory*>(space.get()));
}

TEST(MemorySpaceRegistry, FindMissingSpace)
{
  umpire::space::MemorySpaceRegistry& registry =
    umpire::space::MemorySpaceRegistry::getInstance();

  ASSERT_THROW(registry.getMemorySpaceFactory("UAEUAOEKAJUAOE"),
      umpire::util::Exception);
}
