#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "umpire/strategy/AllocationStrategyRegistry.hpp"
#include "umpire/Allocator.hpp"

using namespace testing;

class MockAllocatorFactory : public umpire::strategy::AllocationStrategyFactory
{
  public:
  MOCK_METHOD1(isValidAllocationStrategyFor, bool(const std::string& name));
  MOCK_METHOD0(create, std::shared_ptr<umpire::strategy::AllocationStrategy>());
  MOCK_METHOD2(createWithTraits, std::shared_ptr<umpire::strategy::AllocationStrategy>(umpire::util::AllocatorTraits, std::vector<std::shared_ptr<umpire::strategy::AllocationStrategy> >));
};

TEST(AllocationStrategyRegistry, Constructor) {
  auto& reg = umpire::strategy::AllocationStrategyRegistry::getInstance();
  SUCCEED();
}

TEST(AllocationStrategyRegistry, Register) {
  auto& reg = umpire::strategy::AllocationStrategyRegistry::getInstance();

  reg.registerAllocationStrategy(std::make_shared<MockAllocatorFactory>());
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
