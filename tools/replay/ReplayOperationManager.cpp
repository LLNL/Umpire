//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <vector>
#include <fstream>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/AllocationPrefetcher.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/ResourceManager.hpp"
#include "ReplayMacros.hpp"
#include "ReplayOperationManager.hpp"

#if !defined(_MSC_VER)
#include <unistd.h>   // getpid()
#else
#include <process.h>
#define getpid _getpid
#endif

ReplayOperationManager::ReplayOperationManager( ReplayFile::Header* Operations ) 
  : m_ops_table{Operations}
{
}

ReplayOperationManager::~ReplayOperationManager() { }

void ReplayOperationManager::printInfo()
{
  std::cout << m_ops_table->num_allocators << " Allocators:" << std::endl;

  for ( std::size_t n{0}; n < m_ops_table->num_allocators; ++n) {
    auto alloc = &m_ops_table->allocators[n];

    std::cout
      << std::setw(6) << std::setfill(' ') << "#"
      << std::setw(2) << std::setfill('0') << n << "  ";

    switch (alloc->type) {
    case ReplayFile::rtype::MEMORY_RESOURCE:
      std::cout << "makeMemoryResource(\"" << m_ops_table->allocators[n].name << "\")";
      break;

    case ReplayFile::rtype::ALLOCATION_ADVISOR:
      std::cout << "<umpire::strategy::AllocationAdvisor, "
        << (alloc->introspection == true ? "true" : "false") << ">"
        << "( \"" << alloc->name << "\""
        << ", \"" << alloc->base_name << "\""
        << ", \"" << alloc->argv.advisor.advice << "\"";

      if (alloc->argv.advisor.device_id >= 0) { // Optional device ID specified
        switch ( alloc->argc ) {
        default:
          REPLAY_ERROR("Invalid number of arguments " << alloc->argc);
          break;
        case 3:
          std::cout << ", " << alloc->argv.advisor.device_id << " )";
          break;
        case 4:
          std::cout << ", \"" << alloc->argv.advisor.accessing_allocator << "\"" << ", " << alloc->argv.advisor.device_id << " )";
          break;
        }
      }
      else { // Use default device_id
        switch ( alloc->argc ) {
        default:
          REPLAY_ERROR("Invalid number of arguments " << alloc->argc);
        case 2:
          std::cout << " )";
          break;
        case 3:
          std::cout << ", \"" << alloc->argv.advisor.accessing_allocator << "\" )";
          break;
        }
      }
      break;

    case ReplayFile::rtype::ALLOCATION_PREFETCHER:
      std::cout << "<umpire::strategy::AllocationPrefetcher, "
        << (alloc->introspection == true ? "true" : "false") << ">"
        << "( " << "\"" << alloc->name << "\"" << ", \"" << alloc->base_name << "\" )";
      break;

    case ReplayFile::rtype::DYNAMIC_POOL_LIST:
      std::cout << "<umpire::strategy::DynamicPoolList, "
        << (alloc->introspection == true ? "true" : "false") << ">"
        << "( " << "\"" << alloc->name << "\""
        << ", \"" << alloc->base_name << "\"";

      switch ( alloc->argc ) {
      default:
        REPLAY_ERROR("Invalid number of arguments " << alloc->argc);
        break;
      case 3:
        std::cout << ", " << alloc->argv.dynamic_pool_list.initial_alloc_size
          << ", " << alloc->argv.dynamic_pool_list.min_alloc_size;
        break;
      case 2:
        std::cout << ", " << alloc->argv.dynamic_pool_list.initial_alloc_size;
        break;
      case 1:
        break;
      }
      std::cout << " )";

      break;

    case ReplayFile::rtype::DYNAMIC_POOL_MAP:
      std::cout << "<umpire::strategy::DynamicPoolMap, "
        << (alloc->introspection == true ? "true" : "false") << ">"
        << "( " << "\"" << alloc->name << "\""
        << ", \"" << alloc->base_name << "\"";

      switch ( alloc->argc ) {
      default:
        REPLAY_ERROR("Invalid number of arguments " << alloc->argc);
        break;
      case 4:
        std::cout << ", " << alloc->argv.dynamic_pool_map.initial_alloc_size
          << ", " << alloc->argv.dynamic_pool_map.min_alloc_size
          << ", " << alloc->argv.dynamic_pool_map.alignment;
        break;
      case 3:
        std::cout << ", " << alloc->argv.dynamic_pool_map.initial_alloc_size
          << ", " << alloc->argv.dynamic_pool_map.min_alloc_size;
        break;
      case 2:
        std::cout << ", " << alloc->argv.dynamic_pool_map.initial_alloc_size;
        break;
      case 1:
        break;
      }

      std::cout << " )";
      break;

    case ReplayFile::rtype::MIXED_POOL:
      std::cout << "<umpire::strategy::MixedPool, "
        << (alloc->introspection == true ? "true" : "false") << ">"
        << "( " << "\"" << alloc->name << "\""
        << ", \"" << alloc->base_name << "\"";

      switch ( alloc->argc ) {
      default:
        REPLAY_ERROR("Invalid number of arguments " << alloc->argc);
        break;
      case 8:
        std::cout
          << ", " << alloc->argv.mixed_pool.smallest_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.largest_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.max_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.size_multiplier
          << ", " << alloc->argv.mixed_pool.dynamic_initial_alloc_bytes
          << ", " << alloc->argv.mixed_pool.dynamic_min_alloc_bytes
          << ", " << alloc->argv.mixed_pool.dynamic_align_bytes;
        break;
      case 7:
        std::cout
          << ", " << alloc->argv.mixed_pool.smallest_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.largest_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.max_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.size_multiplier
          << ", " << alloc->argv.mixed_pool.dynamic_initial_alloc_bytes
          << ", " << alloc->argv.mixed_pool.dynamic_min_alloc_bytes;
        break;
      case 6:
        std::cout
          << ", " << alloc->argv.mixed_pool.smallest_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.largest_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.max_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.size_multiplier
          << ", " << alloc->argv.mixed_pool.dynamic_initial_alloc_bytes;
        break;
      case 5:
        std::cout
          << ", " << alloc->argv.mixed_pool.smallest_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.largest_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.max_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.size_multiplier;
        break;
      case 4:
        std::cout
          << ", " << alloc->argv.mixed_pool.smallest_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.largest_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.max_fixed_blocksize;
        break;
      case 3:
        std::cout
          << ", " << alloc->argv.mixed_pool.smallest_fixed_blocksize
          << ", " << alloc->argv.mixed_pool.largest_fixed_blocksize;
        break;
      case 2:
        std::cout << ", " << alloc->argv.mixed_pool.smallest_fixed_blocksize;
        break;
      case 1:
        break;
      }

      std::cout << " )";
      break;

    case ReplayFile::rtype::MONOTONIC:
      std::cout << "<umpire::strategy::MonotonicAllocationStrategy, "
        << (alloc->introspection == true ? "true" : "false") << ">"
        << "( " << "\"" << alloc->name << "\""
        << ", \"" << alloc->base_name << "\""
        << ", " << alloc->argv.monotonic_pool.capacity << " )";

      break;

    case ReplayFile::rtype::SLOT_POOL:
      std::cout << "<umpire::strategy::SlotPool, "
        << (alloc->introspection == true ? "true" : "false") << ">"
        << "( " << "\"" << alloc->name << "\""
        << ", \"" << alloc->base_name << "\""
        << ", " << alloc->argv.slot_pool.slots << " )";
      break;

    case ReplayFile::rtype::SIZE_LIMITER:
      std::cout << "<umpire::strategy::SizeLimiter, "
        << (alloc->introspection == true ? "true" : "false") << ">"
        << "( " << "\"" << alloc->name << "\""
        << ", \"" << alloc->base_name << "\""
        << ", " << alloc->argv.size_limiter.size_limit << " )";
      break;

    case ReplayFile::rtype::THREADSAFE_ALLOCATOR:
      std::cout << "<umpire::strategy::ThreadSafeAllocator, "
        << (alloc->introspection == true ? "true" : "false") << ">"
        << "( " << "\"" << alloc->name << "\""
        << ", \"" << alloc->base_name << "\"" << " )";
      break;

    case ReplayFile::rtype::FIXED_POOL:
      std::cout << "<umpire::strategy::FixedPool, "
        << (alloc->introspection == true ? "true" : "false") << ">"
        << "( "
        << "\"" << alloc->name << "\""
        << ", \"" << alloc->base_name << "\""
        << ", " << alloc->argv.fixed_pool.object_bytes;

      switch ( alloc->argc ) {
      default:
        REPLAY_ERROR("Invalid number of arguments " << alloc->argc);
        break;
      case 3:
        std::cout << ", " << alloc->argv.fixed_pool.objects_per_pool << " )";
        break;
      case 2:
        std::cout << " )";
        break;
      }
      break;

    default:
      REPLAY_ERROR("Unknown allocator type: " << alloc->type);
      break;
    }

    std::cout << std::endl;
  }
  std::cout << m_ops_table->num_operations << " Operations" << std::endl;
}

void ReplayOperationManager::runOperations(bool gather_statistics)
{
  std::size_t op_counter{0};
  auto& rm = umpire::ResourceManager::getInstance();

  for ( auto op = &m_ops_table->ops[0];
        op < &m_ops_table->ops[m_ops_table->num_operations];
        ++op)
  {
    switch (op->type) {
      case ReplayFile::otype::ALLOCATOR_CREATION:
        makeAllocator(op);
        break;
      case ReplayFile::otype::ALLOCATE:
        makeAllocate(op);
        break;
      case ReplayFile::otype::DEALLOCATE:
        makeDeallocate(op);
        break;
      case ReplayFile::otype::COALESCE:
        makeCoalesce(op);
        break;
      case ReplayFile::otype::RELEASE:
        makeRelease(op);
        break;
      default:
        REPLAY_ERROR("Unknown operation type: " << op->type);
        break;
    }

    if (gather_statistics) {
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

  if (gather_statistics) {
    dumpStats();
  }
}

void ReplayOperationManager::makeAllocator(ReplayFile::Operation* op)
{
  auto alloc = &m_ops_table->allocators[op->allocator_table_index];
  auto& rm = umpire::ResourceManager::getInstance();

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

  case ReplayFile::rtype::DYNAMIC_POOL_LIST:
    if (alloc->argc >= 3) {
      if (alloc->introspection) {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolList, true>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.dynamic_pool_list.initial_alloc_size
              , alloc->argv.dynamic_pool_list.min_alloc_size
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolList, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.dynamic_pool_list.initial_alloc_size
              , alloc->argv.dynamic_pool_list.min_alloc_size
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
              , alloc->argv.dynamic_pool_list.initial_alloc_size
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolList, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.dynamic_pool_list.initial_alloc_size
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
              , alloc->argv.dynamic_pool_map.initial_alloc_size
              , alloc->argv.dynamic_pool_map.min_alloc_size
              , alloc->argv.dynamic_pool_map.alignment
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolMap, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.dynamic_pool_list.initial_alloc_size
              , alloc->argv.dynamic_pool_list.min_alloc_size
              , alloc->argv.dynamic_pool_map.alignment
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
              , alloc->argv.dynamic_pool_map.initial_alloc_size
              , alloc->argv.dynamic_pool_map.min_alloc_size
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolMap, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.dynamic_pool_list.initial_alloc_size
              , alloc->argv.dynamic_pool_list.min_alloc_size
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
              , alloc->argv.dynamic_pool_map.initial_alloc_size
            )
        );
      }
      else {
        alloc->allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::DynamicPoolMap, false>
            (   alloc->name
              , rm.getAllocator(alloc->base_name)
              , alloc->argv.dynamic_pool_list.initial_alloc_size
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
    REPLAY_ERROR("Unknown operation type: " << op->type);
    break;
  }
}

void ReplayOperationManager::makeAllocate(ReplayFile::Operation* op)
{
  auto alloc = &m_ops_table->allocators[op->allocator_table_index];
  op->argv.allocate.ptr = alloc->allocator->allocate(op->argv.allocate.size);
}

void ReplayOperationManager::makeDeallocate(ReplayFile::Operation* op)
{
  auto alloc = &m_ops_table->allocators[op->allocator_table_index];
  auto ptr = m_ops_table->ops[op->argv.deallocate.allocation_op_idx].argv.allocate.ptr;
  alloc->allocator->deallocate(ptr);
}

void ReplayOperationManager::makeCoalesce(ReplayFile::Operation* op)
{
  auto alloc = &m_ops_table->allocators[op->allocator_table_index];

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
  auto alloc = &m_ops_table->allocators[op->allocator_table_index];
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

