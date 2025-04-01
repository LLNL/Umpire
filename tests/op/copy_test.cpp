//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <cstring>
#include <random>

#include "gtest/gtest.h"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/op.hpp"

TEST(Copy, HostToHost)
{
  constexpr std::size_t size = 1024;
  constexpr int num_elements = size / sizeof(int);

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  // Allocate source and destination buffers
  int* source_ptr = static_cast<int*>(allocator.allocate(size));
  int* dest_ptr = static_cast<int*>(allocator.allocate(size));

  // Fill source with random data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distribution(1, 100);

  for (int i = 0; i < num_elements; ++i) {
    source_ptr[i] = distribution(gen);
  }

  // Set destination to zero
  std::memset(dest_ptr, 0, size);

  // Copy data using umpire::copy
  umpire::copy(source_ptr, dest_ptr, size);

  // Verify the copy was successful
  for (int i = 0; i < num_elements; ++i) {
    ASSERT_EQ(source_ptr[i], dest_ptr[i]) << "Data mismatch at index " << i;
  }

  // Cleanup
  allocator.deallocate(source_ptr);
  allocator.deallocate(dest_ptr);
}

TEST(Copy, TypedHostToHost)
{
  constexpr std::size_t num_elements = 1024;

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  // Allocate source and destination buffers
  float* source_ptr = static_cast<float*>(allocator.allocate(num_elements * sizeof(float)));
  float* dest_ptr = static_cast<float*>(allocator.allocate(num_elements * sizeof(float)));

  // Fill source with test data (index as float)
  for (std::size_t i = 0; i < num_elements; ++i) {
    source_ptr[i] = static_cast<float>(i) + 0.5f;
  }

  // Set destination to zero
  std::memset(dest_ptr, 0, num_elements * sizeof(float));

  // Copy data using umpire::copy
  umpire::copy(source_ptr, dest_ptr, num_elements);

  // Verify the copy was successful
  for (std::size_t i = 0; i < num_elements; ++i) {
    ASSERT_FLOAT_EQ(source_ptr[i], dest_ptr[i]) << "Data mismatch at index " << i;
  }

  // Cleanup
  allocator.deallocate(source_ptr);
  allocator.deallocate(dest_ptr);
}

TEST(Copy, PartialCopy)
{
  constexpr std::size_t size = 1024;
  constexpr std::size_t partial_size = 512;
  constexpr int num_elements = size / sizeof(int);
  constexpr int partial_elements = partial_size / sizeof(int);

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  // Allocate source and destination buffers
  int* source_ptr = static_cast<int*>(allocator.allocate(size));
  int* dest_ptr = static_cast<int*>(allocator.allocate(size));

  // Fill source and destination with known patterns
  for (int i = 0; i < num_elements; ++i) {
    source_ptr[i] = i + 100; // Source pattern
    dest_ptr[i] = i;         // Destination pattern
  }

  // Copy partial data using umpire::copy
  umpire::copy(source_ptr, dest_ptr, partial_size);

  // Verify partial copy was successful
  for (int i = 0; i < partial_elements; ++i) {
    ASSERT_EQ(source_ptr[i], dest_ptr[i]) << "Data mismatch at index " << i;
  }

  // Verify the rest of destination is unchanged
  for (int i = partial_elements; i < num_elements; ++i) {
    ASSERT_EQ(dest_ptr[i], i) << "Data unexpectedly changed at index " << i;
  }

  // Cleanup
  allocator.deallocate(source_ptr);
  allocator.deallocate(dest_ptr);
}

// Test for zero-sized copy
TEST(Copy, ZeroSize)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  // Allocate source and destination buffers
  void* source_ptr = allocator.allocate(1024);
  void* dest_ptr = allocator.allocate(1024);

  // Fill with known values
  std::memset(source_ptr, 0xAA, 1024);
  std::memset(dest_ptr, 0xBB, 1024);

  // Copy with zero size - should be a no-op
  umpire::copy(source_ptr, dest_ptr, 0);

  // Check first byte to make sure it wasn't changed
  ASSERT_EQ(*static_cast<unsigned char*>(dest_ptr), 0xBB);

  // Cleanup
  allocator.deallocate(source_ptr);
  allocator.deallocate(dest_ptr);
}

// Additional tests for async copy operations could be added here
