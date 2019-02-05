//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/Macros.hpp"
#include "umpire/strategy/MixedPool.hpp"

namespace umpire { 
namespace strategy {

namespace {
struct data_64_t{ char _[64]; };
struct data_128_t{ char _[128]; };
struct data_256_t{ char _[256]; };
struct data_512_t{ char _[512]; };
struct data_1024_t{ char _[1024]; }; 
struct data_2048_t{ char _[2048]; };
struct data_4096_t{ char _[4096]; };
struct data_8192_t{ char _[8192]; };
struct data_16384_t{ char _[16384]; };
struct data_32768_t{ char _[32768]; };
struct data_131072_t{ char _[131072]; };
struct data_262144_t{ char _[262144]; };
struct data_524288_t{ char _[524288]; };
struct data_1048576_t{ char _[1048576]; };
struct data_2097152_t{ char _[2097152]; };
struct data_4194304_t{ char _[4194304]; };
}

MixedPool::MixedPool(
    const std::string& name,
    int id,
    Allocator allocator) noexcept
:
  AllocationStrategy(name, id),
  m_allocator(allocator.getAllocationStrategy())
{
  m_dynamic_pool = std::make_shared<DynamicPool>(
      "internal_dynamic_pool",
      -1,
      allocator);
  
  m_fixed_pool[0] = std::make_shared<FixedPool<data_64_t> >(
      "internal_fixed_64",
      -1,
      allocator);

  m_fixed_pool[1] = std::make_shared<FixedPool<data_128_t> >(
      "internal_fixed_128",
      -1,
      allocator);

  m_fixed_pool[2] = std::make_shared<FixedPool<data_256_t> >(
      "internal_fixed_256",
      -1,
      allocator);

  m_fixed_pool[3] = std::make_shared<FixedPool<data_512_t> >(
      "internal_fixed_512",
      -1,
      allocator);

  m_fixed_pool[4] = std::make_shared<FixedPool<data_1024_t> >(
      "internal_fixed_1024",
      -1,
      allocator);

  m_fixed_pool[5] = std::make_shared<FixedPool<data_2048_t> >(
      "internal_fixed_2048",
      -1,
      allocator);

  m_fixed_pool[6] = std::make_shared<FixedPool<data_4096_t> >(
      "internal_fixed_4096",
      -1,
      allocator);

  m_fixed_pool[7] = std::make_shared<FixedPool<data_8192_t> >(
      "internal_fixed_8192",
      -1,
      allocator);

  m_fixed_pool[8] = std::make_shared<FixedPool<data_16384_t> >(
      "internal_fixed_16384",
      -1,
      allocator);

  m_fixed_pool[9] = std::make_shared<FixedPool<data_32768_t> >(
      "internal_fixed_32768",
      -1,
      allocator);

  m_fixed_pool[10] = std::make_shared<FixedPool<data_131072_t> >(
      "internal_fixed_131072",
      -1,
      allocator);

  m_fixed_pool[11] = std::make_shared<FixedPool<data_262144_t> >(
      "internal_fixed_262144",
      -1,
      allocator);

  m_fixed_pool[12] = std::make_shared<FixedPool<data_524288_t> >(
      "internal_fixed_524288",
      -1,
      allocator);

  m_fixed_pool[13] = std::make_shared<FixedPool<data_1048576_t> >(
      "internal_fixed_1048576",
      -1,
      allocator);

  m_fixed_pool[14] = std::make_shared<FixedPool<data_2097152_t> >(
      "internal_fixed_2097152",
      -1,
      allocator);

  m_fixed_pool[15] = std::make_shared<FixedPool<data_4194304_t> >(
      "internal_fixed_4194304",
      -1,
      allocator);
}


void* MixedPool::allocate(size_t bytes)
{
  size_t nearest = 1;

  size_t original_bytes = bytes;

  if (bytes <= 1) {
    nearest = 1;
  } else {
    nearest = 2;
    bytes--;
    while (bytes >>= 1) nearest <<= 1;
  }

  (void) original_bytes;

  //std::cout << nearest << std::endl;

  switch (nearest) {
    case 64: return m_fixed_pool[0]->allocate(nearest);
    case 128: return  m_fixed_pool[1]->allocate(nearest);
    case 256: return m_fixed_pool[2]->allocate(nearest);
    case 512: return m_fixed_pool[3]->allocate(nearest);
    case 1024: return m_fixed_pool[4]->allocate(nearest);
    case 2048: return m_fixed_pool[5]->allocate(nearest);
    case 4096: return m_fixed_pool[6]->allocate(nearest);
    case 8192: return m_fixed_pool[7]->allocate(nearest);
    case 16384: return m_fixed_pool[8]->allocate(nearest);
    case 32768: return m_fixed_pool[9]->allocate(nearest);
    case 131072: return m_fixed_pool[10]->allocate(nearest);
    case 262144: return m_fixed_pool[11]->allocate(nearest);
    case 524288: return m_fixed_pool[12]->allocate(nearest);
    case 1048576: return m_fixed_pool[13]->allocate(nearest);
    case 2097152: return m_fixed_pool[14]->allocate(nearest);
    case 4194304: return m_fixed_pool[15]->allocate(nearest);
    default: return m_dynamic_pool->allocate(original_bytes);
  }

  return nullptr;
}

void MixedPool::deallocate(void*)
{
}

void MixedPool::release()
{
}

long 
MixedPool::getCurrentSize() const noexcept
{
  return 0;
}

long 
MixedPool::getActualSize() const noexcept
{
  return 0;
}

long MixedPool::getHighWatermark() const noexcept
{
  return 0;
}

Platform 
MixedPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

}
}
