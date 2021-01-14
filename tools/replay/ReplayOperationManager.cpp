//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/AllocationPrefetcher.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/wrap_allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "ReplayMacros.hpp"
#include "ReplayOperationManager.hpp"
#include "ReplayOptions.hpp"

#include "ReplayOptions.hpp"
#if defined(UMPIRE_ENABLE_NUMA)
#include "umpire/strategy/NumaPolicy.hpp"
#include "umpire/util/numa.hpp"
#endif // defined(UMPIRE_ENABLE_NUMA)

#if !defined(_MSC_VER)
#include <unistd.h>   // getpid()
#else
#include <process.h>
#define getpid _getpid
#endif

ReplayOperationManager::ReplayOperationManager( const ReplayOptions& options,
  ReplayFile* rFile, ReplayFile::Header* Operations )
    : m_options{options}, m_replay_file{rFile}, m_ops_table{Operations}
{
}

ReplayOperationManager::~ReplayOperationManager()
{
  for (std::size_t i = 0; i < m_ops_table->num_allocators; i++) {
    auto alloc = &m_ops_table->allocators[i];
    if (alloc->allocator != nullptr)
      delete(alloc->allocator);
  }
}

namespace {
  struct TrackedCounter {
    void increment() {
      current_count++;
      if (current_count > high_watermark)
        high_watermark = current_count;
    };

    void decrement() {
      current_count--;
    };

    std::size_t current_count{0};
    std::size_t high_watermark{0};
  };

  //
  // https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers
  //
  const int tab64[64] = {
    63,  0, 58,  1, 59, 47, 53,  2,
    60, 39, 48, 27, 54, 33, 42,  3,
    61, 51, 37, 40, 49, 18, 28, 20,
    55, 30, 34, 11, 43, 14, 22,  4,
    62, 57, 46, 52, 38, 26, 32, 41,
    50, 36, 17, 19, 29, 10, 13, 21,
    56, 45, 25, 31, 35, 16,  9, 12,
    44, 24, 15,  8, 23,  7,  6,  5};

  int log2_64 (std::size_t size)
  {
    uint64_t value{static_cast<uint64_t>(size)};
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    uint64_t multiplier = UINT64_C(0x07EDD5E59A4E28C2);
    uint64_t shifter = UINT64_C(58);
    value = value - (value >> 1);
    int index = (value * multiplier) >> shifter;
    return tab64[index];
  }

  struct TrackedHistogram {
    void increment(std::size_t size) {
      int index{ log2_64(size) };
      log2_buckets[index].increment();
    };

    void decrement(std::size_t size) {
      int index{ log2_64(size) };
      log2_buckets[index].decrement();
    };

    void print() const {
      std::cout << log2_buckets[0].high_watermark;
      for ( int i = 1; i < 64; i++ ) {
        std::cout << ", " << log2_buckets[i].high_watermark;
      }
      std::cout << std::endl;
    };

    TrackedCounter log2_buckets[64]{};
  };
}

void ReplayOperationManager::runOperations()
{
  std::map<int, TrackedHistogram > size_histogram;
  std::size_t op_counter{0};
  auto& rm = umpire::ResourceManager::getInstance();

  if (m_options.print_stats_on_release) {
    std::cout << "Input,Release,Name,CurrentSize,ActualSize,Watermark"
      << std::endl;
  }

  for ( auto op = &m_ops_table->ops[1];
        op < &m_ops_table->ops[m_ops_table->num_operations];
        ++op)
  {
    try {
      switch (op->op_type) {
        case ReplayFile::otype::ALLOCATOR_CREATION:
          if (m_options.print_stats_on_release) {
            size_histogram[op->op_allocator] = TrackedHistogram{};
          }
          makeAllocator(op);
          break;
        case ReplayFile::otype::SETDEFAULTALLOCATOR:
          if (m_options.print_stats_on_release) {
            size_histogram[op->op_allocator] = TrackedHistogram{};
          }
          makeSetDefaultAllocator(op);
          break;
        case ReplayFile::otype::COPY:
          if (m_options.skip_operations == false) {
            makeCopy(op);
          }
          break;
        case ReplayFile::otype::REALLOCATE:
          makeReallocate(op);
          break;
        case ReplayFile::otype::REALLOCATE_EX:
          makeReallocate_ex(op);
          break;
        case ReplayFile::otype::ALLOCATE:
          if (m_options.print_stats_on_release) {
            size_histogram[op->op_allocator].increment(op->op_size);
          }
          makeAllocate(op);
          break;
        case ReplayFile::otype::DEALLOCATE:
          if (m_options.print_stats_on_release) {
            auto alloc = &m_ops_table->allocators[op->op_allocator];
            auto ptr = m_ops_table->ops[op->op_alloc_ops[0]].op_allocated_ptr;
            size_histogram[op->op_allocator].decrement(
                                                alloc->allocator->getSize(ptr));
          }
          makeDeallocate(op);
          break;
        case ReplayFile::otype::COALESCE:
          makeCoalesce(op);
          break;
        case ReplayFile::otype::RELEASE:
          makeRelease(op);
          break;
        default:
          REPLAY_ERROR("Unknown operation type: " << op->op_type);
          break;
      }
    }
    catch(...) {
      std::cerr << std::endl << std::endl
        << "Replay Failure Line Number: " << std::endl
        << "  Line: " << op->op_line_number << m_replay_file->getLine(op->op_line_number)
        << std::endl << std::endl;
      throw;
    }

    if (m_options.print_statistics) {
      for (const auto& alloc_name : rm.getAllocatorNames()) {
        auto alloc = rm.getAllocator(alloc_name);

        std::string cur_stat_name{alloc_name + " current_size"};
        std::string actual_stat_name{alloc_name + " actual_size"};
        std::string hwm_stat_name{alloc_name + " hwm"};

        m_stat_series[cur_stat_name].push_back(
            std::make_pair(
              op_counter,
              alloc.getCurrentSize()));

        m_stat_series[actual_stat_name].push_back(
            std::make_pair(
              op_counter,
              alloc.getActualSize()));

        m_stat_series[hwm_stat_name].push_back(
            std::make_pair(
              op_counter,
              alloc.getHighWatermark()));
      }
    }
    op_counter++;
  }

  if (m_options.print_statistics) {
    dumpStats();
  }

  if (m_options.print_stats_on_release) {
    for (const auto& alloc_name : rm.getAllocatorNames()) {
      auto alloc = rm.getAllocator(alloc_name);
      std::cout << m_replay_file->getInputFileName() << ","
              << "End,"
              << alloc_name << ","
              << alloc.getCurrentSize() << ","
              << alloc.getActualSize() << ","
              << alloc.getHighWatermark()
              << std::endl;
    }

    for (auto const& x : size_histogram)
    {
      auto alloc = &m_ops_table->allocators[x.first];
      std::cout << alloc->allocator->getName() << ", ";
      x.second.print();
    }
  }
}

void ReplayOperationManager::makeAllocator(ReplayFile::Operation* op)
{
  auto alloc = &m_ops_table->allocators[op->op_allocator];
  auto& rm = umpire::ResourceManager::getInstance();

  //
  // Check to see if user requested that we switch to a different pool
  //
  if ( !m_options.pool_to_use.empty() ) {
    if (   alloc->type == ReplayFile::rtype::DYNAMIC_POOL_LIST
        || alloc->type == ReplayFile::rtype::DYNAMIC_POOL_MAP
        || alloc->type == ReplayFile::rtype::QUICKPOOL) {
      if (m_options.pool_to_use == "List") {
        alloc->type = ReplayFile::rtype::DYNAMIC_POOL_LIST;
      }
      else if (m_options.pool_to_use == "Map") {
        alloc->type = ReplayFile::rtype::DYNAMIC_POOL_MAP;
      }
      else if (m_options.pool_to_use == "Quick") {
        alloc->type = ReplayFile::rtype::QUICKPOOL;
      }
    }
  }

  switch (alloc->type) {
  case ReplayFile::rtype::MEMORY_RESOURCE:
    alloc->allocator = new umpire::Allocator(rm.getAllocator(alloc->name));
    break;

  case ReplayFile::rtype::ALLOCATION_ADVISOR:
    if (alloc->argv.advisor.device_id >= 0) { // Optional device ID specified
      switch ( alloc->argc ) {
      default:
        REPLAY_ERROR("Invalid number of arguments " << alloc->argc);
      case 3:
        if (alloc->introspection) {
          alloc->allocator = new umpire::Allocator(
            rm.makeAllocator<umpire::strategy::AllocationAdvisor, true>
              (   alloc->name
                , rm.getAllocator(alloc->base_name)
                , alloc->argv.advisor.advice
                , alloc->argv.advisor.device_id
              )
          );
        }
        else {
          alloc->allocator = new umpire::Allocator(
            rm.makeAllocator<umpire::strategy::AllocationAdvisor, false>
              (   alloc->name
                , rm.getAllocator(alloc->base_name)
                , alloc->argv.advisor.advice
                , alloc->argv.advisor.device_id
              )
          );
        }
        break;
      case 4:
        if (alloc->introspection) {
          alloc->allocator = new umpire::Allocator(
            rm.makeAllocator<umpire::strategy::AllocationAdvisor, true>
              (   alloc->name
                , rm.getAllocator(alloc->base_name)
                , alloc->argv.advisor.advice
                , rm.getAllocator(alloc->argv.advisor.accessing_allocator)
                , alloc->argv.advisor.device_id
              )
          );
        }
        else {
          alloc->allocator = new umpire::Allocator(
            rm.makeAllocator<umpire::strategy::AllocationAdvisor, false>
              (   alloc->name
                , rm.getAllocator(alloc->base_name)
                , alloc->argv.advisor.advice
                , rm.getAllocator(alloc->argv.advisor.accessing_allocator)
                , alloc->argv.advisor.device_id
              )
          );
        }
        break;
      }
    }
    else { // Use default device_id
      switch ( alloc->argc ) {
      default:
        REPLAY_ERROR("Invalid number of arguments " << alloc->argc);
      case 2:
        if (alloc->introspection) {
          alloc->allocator = new umpire::Allocator(
            rm.makeAllocator<umpire::strategy::AllocationAdvisor, true>
              (   alloc->name
                , rm.getAllocator(alloc->base_name)
                , alloc->argv.advisor.advice
              )
          );
        }
        else {
          alloc->allocator = new umpire::Allocator(
            rm.makeAllocator<umpire::strategy::AllocationAdvisor, false>
              (   alloc->name
                , rm.getAllocator(alloc->base_name)
                , alloc->argv.advisor.advice
              )
          );
        }
        break;
      case 3:
        if (alloc->introspection) {
          alloc->allocator = new umpire::Allocator(
            rm.makeAllocator<umpire::strategy::AllocationAdvisor, true>
              (   alloc->name
                , rm.getAllocator(alloc->base_name)
                , alloc->argv.advisor.advice
                , rm.getAllocator(alloc->argv.advisor.accessing_allocator)
              )
          );
        }
        else {
          alloc->allocator = new umpire::Allocator(
            rm.makeAllocator<umpire::strategy::AllocationAdvisor, false>
              (   alloc->name
                , rm.getAllocator(alloc->base_name)
                , alloc->argv.advisor.advice
                , rm.getAllocator(alloc->argv.advisor.accessing_allocator)
              )
          );
        }
        break;
      }
    }
    break;

  case ReplayFile::rtype::ALLOCATION_PREFETCHER:
    if (alloc->introspection) {
      alloc->allocator = new umpire::Allocator(
        rm.makeAllocator<umpire::strategy::AllocationPrefetcher, true>
          (   alloc->name
            , rm.getAllocator(alloc->base_name)
          )
      );
    }
    else {
      alloc->allocator = new umpire::Allocator(
        rm.makeAllocator<umpire::strategy::AllocationPrefetcher, false>
          (   alloc->name
            , rm.getAllocator(alloc->base_name)
          )
      );
    }
    break;

  case ReplayFile::rtype::NUMA_POLICY:
#if defined(UMPIRE_ENABLE_NUMA)
    if (alloc->introspection) {
      alloc->allocator = new umpire::Allocator(
        rm.makeAllocator<umpire::strategy::NumaPolicy, true>
          (   alloc->name
            , rm.getAllocator(alloc->base_name)
            , alloc->argv.numa.node
          )
      );
    }
    else {
      alloc->allocator = new umpire::Allocator(
        rm.makeAllocator<umpire::strategy::NumaPolicy, false>
          (   alloc->name
            , rm.getAllocator(alloc->base_name)
            , alloc->argv.numa.node
          )
      );
    }
#else
    std::cerr
      << "Warning, NUMA policy operation found and skipped, consider building"
      << std::endl
      << "version of replay with -DENABLE_NUMA=On."
      << std::endl;
#endif // defined(UMPIRE_ENABLE_NUMA)
    break;

  case ReplayFile::rtype::QUICKPOOL:
    if (alloc->argc >= 4) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::QuickPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
              , alloc->argv.pool.min_alloc_size
              , alloc->argv.pool.alignment
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::QuickPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
              , alloc->argv.pool.min_alloc_size
              , alloc->argv.pool.alignment
            )
        );
      }
    }
    else if (alloc->argc == 3) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::QuickPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
              , alloc->argv.pool.min_alloc_size
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::QuickPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
              , alloc->argv.pool.min_alloc_size
            )
        );
      }
    }
    else if (alloc->argc == 2) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::QuickPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::QuickPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
            )
        );
      }
    }
    else {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::QuickPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::QuickPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
            )
        );
      }
    }
    break;

  case ReplayFile::rtype::DYNAMIC_POOL_LIST:
    if (alloc->argc >= 4) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolList, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
              , alloc->argv.pool.min_alloc_size
              , alloc->argv.pool.alignment
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolList, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
              , alloc->argv.pool.min_alloc_size
              , alloc->argv.pool.alignment
            )
        );
      }
    }
    else if (alloc->argc == 3) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolList, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
              , alloc->argv.pool.min_alloc_size
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolList, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
              , alloc->argv.pool.min_alloc_size
            )
        );
      }
    }
    else if (alloc->argc == 2) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolList, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolList, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
            )
        );
      }
    }
    else {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolList, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolList, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
            )
        );
      }
    }
    break;

  case ReplayFile::rtype::DYNAMIC_POOL_MAP:
    if (alloc->argc >= 4) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolMap, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
              , alloc->argv.pool.min_alloc_size
              , alloc->argv.pool.alignment
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolMap, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
              , alloc->argv.pool.min_alloc_size
              , alloc->argv.pool.alignment
            )
        );
      }
    }
    else if (alloc->argc >= 3) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolMap, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
              , alloc->argv.pool.min_alloc_size
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolMap, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
              , alloc->argv.pool.min_alloc_size
            )
        );
      }
    }
    else if (alloc->argc == 2) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolMap, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolMap, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.pool.initial_alloc_size
            )
        );
      }
    }
    else {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolMap, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolMap, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
            )
        );
      }
    }
    break;

  case ReplayFile::rtype::MONOTONIC:
    if (alloc->introspection) {
      alloc->allocator = new umpire::Allocator(
        rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy, true>
          (   alloc->name
            , rm.getAllocator(alloc->base_name)
            , alloc->argv.monotonic_pool.capacity
          )
      );
    }
    else {
      alloc->allocator = new umpire::Allocator(
        rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy, false>
          (   alloc->name
            , rm.getAllocator(alloc->base_name)
            , alloc->argv.monotonic_pool.capacity
          )
      );
    }
    break;

  case ReplayFile::rtype::SLOT_POOL:
    if (alloc->introspection) {
      alloc->allocator = new umpire::Allocator(
        rm.makeAllocator<umpire::strategy::SlotPool, true>
          (   alloc->name
            , rm.getAllocator(alloc->base_name)
            , alloc->argv.slot_pool.slots
          )
      );
    }
    else {
      alloc->allocator = new umpire::Allocator(
        rm.makeAllocator<umpire::strategy::SlotPool, false>
          (   alloc->name
            , rm.getAllocator(alloc->base_name)
            , alloc->argv.slot_pool.slots
          )
      );
    }
    break;

  case ReplayFile::rtype::SIZE_LIMITER:
    if (alloc->introspection) {
      alloc->allocator = new umpire::Allocator(
        rm.makeAllocator<umpire::strategy::SizeLimiter, true>
          (   alloc->name
            , rm.getAllocator(alloc->base_name)
            , alloc->argv.size_limiter.size_limit
          )
      );
    }
    else {
      alloc->allocator = new umpire::Allocator(
        rm.makeAllocator<umpire::strategy::SizeLimiter, false>
          (   alloc->name
            , rm.getAllocator(alloc->base_name)
            , alloc->argv.size_limiter.size_limit
          )
      );
    }
    break;

  case ReplayFile::rtype::THREADSAFE_ALLOCATOR:
    if (alloc->introspection) {
      alloc->allocator = new umpire::Allocator(
        rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, true>
          (   alloc->name
            , rm.getAllocator(alloc->base_name)
          )
      );
    }
    else {
      alloc->allocator = new umpire::Allocator(
        rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, false>
          (   alloc->name
            , rm.getAllocator(alloc->base_name)
          )
      );
    }
    break;

  case ReplayFile::rtype::FIXED_POOL:
    if (alloc->argc >= 3) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::FixedPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.fixed_pool.object_bytes
              , alloc->argv.fixed_pool.objects_per_pool
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::FixedPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.fixed_pool.object_bytes
              , alloc->argv.fixed_pool.objects_per_pool
            )
        );
      }
    }
    else {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::FixedPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.fixed_pool.object_bytes
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::FixedPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.fixed_pool.object_bytes
            )
        );
      }
    }
    break;

  case ReplayFile::rtype::MIXED_POOL:
    if (alloc->argc >= 8) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
              , alloc->argv.mixed_pool.largest_fixed_blocksize
              , alloc->argv.mixed_pool.max_fixed_blocksize
              , alloc->argv.mixed_pool.size_multiplier
              , alloc->argv.mixed_pool.dynamic_initial_alloc_bytes
              , alloc->argv.mixed_pool.dynamic_min_alloc_bytes
              , alloc->argv.mixed_pool.dynamic_align_bytes
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
              , alloc->argv.mixed_pool.largest_fixed_blocksize
              , alloc->argv.mixed_pool.max_fixed_blocksize
              , alloc->argv.mixed_pool.size_multiplier
              , alloc->argv.mixed_pool.dynamic_initial_alloc_bytes
              , alloc->argv.mixed_pool.dynamic_min_alloc_bytes
              , alloc->argv.mixed_pool.dynamic_align_bytes
            )
        );
      }
    }
    else if (alloc->argc >= 7) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
              , alloc->argv.mixed_pool.largest_fixed_blocksize
              , alloc->argv.mixed_pool.max_fixed_blocksize
              , alloc->argv.mixed_pool.size_multiplier
              , alloc->argv.mixed_pool.dynamic_initial_alloc_bytes
              , alloc->argv.mixed_pool.dynamic_min_alloc_bytes
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
              , alloc->argv.mixed_pool.largest_fixed_blocksize
              , alloc->argv.mixed_pool.max_fixed_blocksize
              , alloc->argv.mixed_pool.size_multiplier
              , alloc->argv.mixed_pool.dynamic_initial_alloc_bytes
              , alloc->argv.mixed_pool.dynamic_min_alloc_bytes
            )
        );
      }
    }
    else if (alloc->argc >= 6) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
              , alloc->argv.mixed_pool.largest_fixed_blocksize
              , alloc->argv.mixed_pool.max_fixed_blocksize
              , alloc->argv.mixed_pool.size_multiplier
              , alloc->argv.mixed_pool.dynamic_initial_alloc_bytes
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
              , alloc->argv.mixed_pool.largest_fixed_blocksize
              , alloc->argv.mixed_pool.max_fixed_blocksize
              , alloc->argv.mixed_pool.size_multiplier
              , alloc->argv.mixed_pool.dynamic_initial_alloc_bytes
            )
        );
      }
    }
    else if (alloc->argc >= 5) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
              , alloc->argv.mixed_pool.largest_fixed_blocksize
              , alloc->argv.mixed_pool.max_fixed_blocksize
              , alloc->argv.mixed_pool.size_multiplier
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
              , alloc->argv.mixed_pool.largest_fixed_blocksize
              , alloc->argv.mixed_pool.max_fixed_blocksize
              , alloc->argv.mixed_pool.size_multiplier
            )
        );
      }
    }
    else if (alloc->argc >= 4) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
              , alloc->argv.mixed_pool.largest_fixed_blocksize
              , alloc->argv.mixed_pool.max_fixed_blocksize
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
              , alloc->argv.mixed_pool.largest_fixed_blocksize
              , alloc->argv.mixed_pool.max_fixed_blocksize
            )
        );
      }
    }
    else if (alloc->argc >= 3) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
              , alloc->argv.mixed_pool.largest_fixed_blocksize
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
              , alloc->argv.mixed_pool.largest_fixed_blocksize
            )
        );
      }
    }
    else if (alloc->argc >= 2) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.mixed_pool.smallest_fixed_blocksize
            )
        );
      }
    }
    else {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MixedPool, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
            )
        );
      }
    }
    break;

  default:
    REPLAY_ERROR("Unknown operation type: " << op->op_type);
    break;
  }
}

void ReplayOperationManager::makeAllocate(ReplayFile::Operation* op)
{
  auto alloc = &m_ops_table->allocators[op->op_allocator];

  op->op_allocated_ptr = alloc->allocator->allocate(op->op_size);
}

void ReplayOperationManager::makeSetDefaultAllocator(ReplayFile::Operation* op)
{
  auto alloc = &m_ops_table->allocators[op->op_allocator];
  auto& rm = umpire::ResourceManager::getInstance();
  rm.setDefaultAllocator(*(alloc->allocator));
}

void ReplayOperationManager::makeReallocate(ReplayFile::Operation* op)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto ptr = m_ops_table->ops[op->op_alloc_ops[1]].op_allocated_ptr;
  op->op_allocated_ptr = rm.reallocate(ptr, op->op_size);
}

void ReplayOperationManager::makeReallocate_ex(ReplayFile::Operation* op)
{
  auto alloc = &m_ops_table->allocators[op->op_allocator];
  auto& rm = umpire::ResourceManager::getInstance();
  auto ptr = m_ops_table->ops[op->op_alloc_ops[1]].op_allocated_ptr;
  op->op_allocated_ptr = rm.reallocate(ptr, op->op_size, *(alloc->allocator));
}

void ReplayOperationManager::makeCopy(ReplayFile::Operation* op)
{
  auto& rm = umpire::ResourceManager::getInstance();
  char* src_ptr = static_cast<char*>(m_ops_table->ops[op->op_alloc_ops[0]].op_allocated_ptr);
  char* dst_ptr = static_cast<char*>(m_ops_table->ops[op->op_alloc_ops[1]].op_allocated_ptr);
  auto src_off = op->op_offsets[0];
  auto dst_off = op->op_offsets[1];
  auto size = op->op_size;

  rm.copy(dst_ptr+dst_off, src_ptr+src_off, size);
}

void ReplayOperationManager::makeDeallocate(ReplayFile::Operation* op)
{
  try {
    auto alloc = &m_ops_table->allocators[op->op_allocator];
    auto ptr = m_ops_table->ops[op->op_alloc_ops[0]].op_allocated_ptr;
    alloc->allocator->deallocate(ptr);
  }
  catch (...) {
    std::cerr << std::endl
      << "Deallocation Failure Line Number: " << std::endl
      << "  Line: " << op->op_line_number << m_replay_file->getLine(op->op_line_number) << std::endl
      << "  for memory allocation at:" << std::endl
      << "  Line: " << m_ops_table->ops[op->op_alloc_ops[0]].op_line_number
      << m_replay_file->getLine(
              m_ops_table->ops[op->op_alloc_ops[0]].op_line_number)
      << std::endl << std::endl;
    throw;
  }
}

void ReplayOperationManager::makeCoalesce(ReplayFile::Operation* op)
{
  auto alloc = &m_ops_table->allocators[op->op_allocator];

  try {
    auto dynamic_pool =
      umpire::util::unwrap_allocator<umpire::strategy::DynamicPoolMap>(*(alloc->allocator));
    dynamic_pool->coalesce();
  }
  catch(...) {
    auto dynamic_pool =
      umpire::util::unwrap_allocator<umpire::strategy::DynamicPoolList>(*(alloc->allocator));
    dynamic_pool->coalesce();
  }
}

void ReplayOperationManager::makeRelease(ReplayFile::Operation* op)
{
  auto alloc = &m_ops_table->allocators[op->op_allocator];
  if (m_options.print_stats_on_release) {
    std::cout << m_replay_file->getInputFileName() << ","
              << "Pre Release,"
              << alloc->allocator->getName() << ","
              << alloc->allocator->getCurrentSize() << ","
              << alloc->allocator->getActualSize() << ","
              << alloc->allocator->getHighWatermark()
              << std::endl;
  }
  alloc->allocator->release();
}

void ReplayOperationManager::dumpStats()
{
  std::ofstream file;
  const int pid{getpid()};

  const std::string filename{"replay" + std::to_string(pid) + ".ult"};
  file.open(filename);

  for (const auto& stat_series : m_stat_series) {
    file << "# " << stat_series.first << std::endl;
    for (const auto& entry : stat_series.second) {
      int t;
      std::size_t val;
      std::tie(t, val) = entry;

      file
        <<  t
        << " " << val << std::endl;
    }
  }
}

#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
