#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "umpire/strategy/AllocationStrategyRegistry.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/util/Exception.hpp"
#include "umpire/util/Macros.hpp"

using namespace testing;

class MockAllocatorFactory : public umpire::strategy::AllocationStrategyFactory
{
  public:
  MOCK_METHOD1(isValidAllocationStrategyFor, bool(const std::string& name));
  MOCK_METHOD0(create, std::shared_ptr<umpire::strategy::AllocationStrategy>());
  MOCK_METHOD2(createWithTraits, std::shared_ptr<umpire::strategy::AllocationStrategy>(umpire::util::AllocatorTraits, std::vector<std::shared_ptr<umpire::strategy::AllocationStrategy> >));
};

TEST(AllocatorRegistry, Constructor) {
  umpire::AllocatorRegistry& reg = umpire::AllocatorRegistry::getInstance();
  SUCCEED();
}

TEST(AllocatorRegistry, Register) {
  umpire::AllocatorRegistry& reg = umpire::AllocatorRegistry::getInstance();
  reg.registerAllocator(std::make_shared<MockAllocatorFactory>());

  SUCCEED();
}

TEST(AllocationStrategyRegistry, Create) {
  auto mock_allocator_factory = std::make_shared<MockAllocatorFactory>();

  EXPECT_CALL(*mock_allocator_factory, isValidAllocationStrategyFor("test"))
    .Times(1)
    .WillOnce(Return(true));
  EXPECT_CALL(*mock_allocator_factory, create())
    .Times(1);

  umpire::strategy::AllocationStrategyRegistry reg = umpire::strategy::AllocationStrategyRegistry::getInstance();
  reg.registerAllocationStrategy(mock_allocator_factory);

  auto alloc = reg.makeAllocationStrategy("test", {}, {});
  ASSERT_EQ(std::dynamic_pointer_cast<umpire::strategy::AllocationStrategy>(mock_allocator_factory), alloc);
}

TEST(AllocatorRegistry, CreateUnknown) {
  umpire::AllocatorRegistry& reg = umpire::AllocatorRegistry::getInstance();

  ASSERT_THROW(reg.makeAllocator("unknown"), umpire::util::Exception);
}
