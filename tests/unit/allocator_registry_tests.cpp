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
  MOCK_METHOD3(createWithTraits, std::shared_ptr<umpire::strategy::AllocationStrategy>(const std::string& name, umpire::util::AllocatorTraits, std::vector<std::shared_ptr<umpire::strategy::AllocationStrategy> >));
};

TEST(AllocatorRegistry, Constructor) {
  umpire::strategy::AllocationStrategyRegistry& reg = umpire::strategy::AllocationStrategyRegistry::getInstance();

  (void) reg;

  SUCCEED();
}

TEST(AllocationStrategyRegistry, RegisterAndCreate) {
  auto mock_allocator_factory = std::make_shared<MockAllocatorFactory>();

  EXPECT_CALL(*mock_allocator_factory, isValidAllocationStrategyFor("test"))
    .Times(1)
    .WillOnce(Return(true));
  EXPECT_CALL(*mock_allocator_factory, isValidAllocationStrategyFor("unknown"))
    .WillRepeatedly(Return(false));
  EXPECT_CALL(*mock_allocator_factory, createWithTraits(_, _, _))
    .Times(1);

  umpire::strategy::AllocationStrategyRegistry& reg = umpire::strategy::AllocationStrategyRegistry::getInstance();
  reg.registerAllocationStrategy(mock_allocator_factory);

  umpire::util::AllocatorTraits traits;

  traits.m_initial_size = 0;
  traits.m_maximum_size = 0;
  traits.m_number_allocations = 0;


  auto alloc = reg.makeAllocationStrategy("test", traits, {});
  ASSERT_EQ(std::dynamic_pointer_cast<umpire::strategy::AllocationStrategy>(mock_allocator_factory), alloc);

  ::testing::Mock::AllowLeak(&(*mock_allocator_factory));
}

TEST(AllocationStrategyRegistry, CreateUnknown) {
  umpire::strategy::AllocationStrategyRegistry& reg = umpire::strategy::AllocationStrategyRegistry::getInstance();

  umpire::util::AllocatorTraits traits;

  traits.m_initial_size = 0;
  traits.m_maximum_size = 0;
  traits.m_number_allocations = 0;

  ASSERT_THROW(reg.makeAllocationStrategy("unknown", traits, {}), umpire::util::Exception);
}
