#include "gtest/gtest.h"

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/util/Exception.hpp"

TEST(Operation, HostToHostCopy)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");

  double* array_one = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array_one[i] = i;
  }

  rm.copy(array_one, array_two);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array_one[i], array_two[i]);
  }
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST(Operation, HostToDeviceToHostCopy)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");
  umpire::Allocator device_allocator = rm.getAllocator("DEVICE");

  double* array_one = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(device_allocator.allocate(100*sizeof(double)));

  double* array_three = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array_one[i] = i;
  }

  rm.copy(array_one, array_two);
  rm.copy(array_two, array_three);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array_one[i], array_three[i]);
  }
}

TEST(Operation, HostToUmCopy)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");
  umpire::Allocator um_allocator = rm.getAllocator("UM");

  double* array_one = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(um_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array_one[i] = i;
  }

  rm.copy(array_one, array_two);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array_one[i], array_two[i]);
  }
}

TEST(Operation, UmToHostCopy)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");
  umpire::Allocator um_allocator = rm.getAllocator("UM");

  double* array_one = static_cast<double*>(um_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array_one[i] = i;
  }

  rm.copy(array_one, array_two);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array_one[i], array_two[i]);
  }
}

TEST(Operation, UmToUmCopy)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator um_allocator = rm.getAllocator("UM");

  double* array_one = static_cast<double*>(um_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(um_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array_one[i] = i;
  }

  rm.copy(array_one, array_two);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array_one[i], array_two[i]);
  }
}

TEST(Operation, UmToDeviceToUmCopy)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator um_allocator = rm.getAllocator("UM");
  umpire::Allocator device_allocator = rm.getAllocator("DEVICE");

  double* array_one = static_cast<double*>(um_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(device_allocator.allocate(100*sizeof(double)));
  double* array_three = static_cast<double*>(um_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array_one[i] = i;
  }

  rm.copy(array_one, array_two);

  rm.copy(array_two, array_three);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array_one[i], array_three[i]);
  }
}
#endif

TEST(Operation, SizeError)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");

  double* array_one = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(host_allocator.allocate(70*sizeof(double)));

  ASSERT_THROW(rm.copy(array_one, array_two), umpire::util::Exception);
}

TEST(Operation, CopyOffset)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator host_allocator = rm.getAllocator("HOST");

  double* array_one = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(host_allocator.allocate(70*sizeof(double)));

  array_one[10] = 3.14;
  array_two[11] = 0.0;

  rm.copy(&array_one[10], &array_two[11], sizeof(double));

  ASSERT_EQ(array_one[10], array_two[11]);
}


TEST(Operation, HostMemset)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator host_allocator = rm.getAllocator("HOST");

  double* array = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  rm.memset(array, 0);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array[i], 0);
  }
}

#if defined(ENABLE_CUDA)
TEST(Operation, DeviceMemset)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");
  umpire::Allocator device_allocator = rm.getAllocator("DEVICE");

  double* h_array = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));
  double* d_array = static_cast<double*>(device_allocator.allocate(100*sizeof(double)));

  rm.memset(d_array, 0);

  rm.copy(d_array, h_array);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(h_array[i], 0);
  }
}

TEST(Operation, UmMemset)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator host_allocator = rm.getAllocator("UM");

  double* array = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  rm.memset(array, 0);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array[i], 0);
  }
}
#endif
