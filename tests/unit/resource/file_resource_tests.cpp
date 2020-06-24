#include "resource_tests.hpp"

#include "umpire/resource/FileMemoryResource.hpp"

#include "resource_tests.hpp"

#include "gtest/gtest.h"


REGISTER_TYPED_TEST_SUITE_P(
    ResourceTest,
    Constructor, Allocate, getCurrentSize, getHighWatermark, getPlatform, getTraits);

INSTANTIATE_TYPED_TEST_SUITE_P(Mmap, ResourceTest, umpire::resource::FileMemoryResource,);
