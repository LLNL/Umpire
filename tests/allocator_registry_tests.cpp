#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "umpire/AllocatorRegistry.hpp"
#include "umpire/Allocator.hpp"

using namespace testing;

class MockAllocatorFactory : public umpire::AllocatorFactory
{
  public:
  MOCK_METHOD1(isValidAllocatorFor, bool(const std::string& name));
  MOCK_METHOD0(create, std::shared_ptr<umpire::Allocator>());
};

TEST(AllocatorRegistry, Constructor) {
  umpire::AllocatorRegistry reg = umpire::AllocatorRegistry::getInstance();
  SUCCEED();
}

TEST(AllocatorRegistry, Register) {
  umpire::AllocatorRegistry reg = umpire::AllocatorRegistry::getInstance();

  reg.registerAllocator(std::make_shared<MockAllocatorFactory>());
}

TEST(AllocatorRegistry, Create) {
  auto mock_allocator_factory = std::make_shared<MockAllocatorFactory>();
  EXPECT_CALL(*mock_allocator_factory, isValidAllocatorFor("test"))
    .Times(1)
    .WillOnce(Return(true));
  EXPECT_CALL(*mock_allocator_factory, create())
    .Times(1);

  umpire::AllocatorRegistry reg = umpire::AllocatorRegistry::getInstance();
  reg.registerAllocator(mock_allocator_factory);

  auto alloc = reg.makeAllocator("test");
  ASSERT_EQ(std::dynamic_pointer_cast<umpire::Allocator>(mock_allocator_factory), alloc);
}
