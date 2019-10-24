//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <cstdint>
#include <vector>
#include <fstream>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/strategy/MixedPool.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"
#include "umpire/util/AllocationMap.hpp"
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

void ReplayOperationManager::makeSizeLimiter(
    const bool introspection
  , const std::string allocator_name
  , const std::string base_allocator_name
  , const std::size_t size_limit
)
{
  m_cont_op = new ReplayOperation;

  if (introspection) {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::SizeLimiter, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , size_limit
          )
      );
    };
  }
  else {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::SizeLimiter, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , size_limit
          )
      );
    };
  }

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAllocate( int allocator_num, std::size_t size )
{
  m_cont_op = new ReplayOperation;
  m_cont_op->op = [=]() {
    m_cont_op->m_allocation_ptr = this->m_allocator_array[allocator_num].allocate(size);
  };
  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAllocateCont( uint64_t allocation_from_log )
{
  m_alloc_operations[allocation_from_log] = m_cont_op;
}

ReplayOperationManager::ReplayOperationManager( void ) {
}

ReplayOperationManager::~ReplayOperationManager() { }

void ReplayOperationManager::runOperations(bool gather_statistics)
{
  std::size_t op_counter{0};

  auto& rm = umpire::ResourceManager::getInstance();

  for ( auto resource_name : m_resource_names )
    m_allocator_array.push_back(rm.getAllocator(resource_name));

  for (auto op : operations) {
    op->op();

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

void ReplayOperationManager::makeMemoryResource( const std::string resource_name )
{
  m_resource_names.push_back(resource_name);
}

void ReplayOperationManager::makeMonotonicAllocator(
    const bool introspection
  , const std::string allocator_name
  , const std::size_t capacity
  , const std::string base_allocator_name
)
{
  m_cont_op = new ReplayOperation;

  if (introspection) {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , capacity
          )
      );
    };
  }
  else {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , capacity
          )
      );
    };
  }

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeSlotPool(
    const bool introspection
  , const std::string allocator_name
  , const std::size_t slots
  , const std::string base_allocator_name
)
{
  m_cont_op = new ReplayOperation;

  if (introspection) {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::SlotPool, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , slots
          )
      );
    };
  }
  else {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::SlotPool, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , slots
          )
      );
    };
  }

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeThreadSafeAllocator(
    const bool introspection
  , const std::string allocator_name
  , const std::string base_allocator_name
)
{
  m_cont_op = new ReplayOperation;

  if (introspection) {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
          )
      );
    };
  }
  else {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
          )
      );
    };
  }

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAllocatorCont( void )
{
}

void ReplayOperationManager::makeDeallocate( int allocator_num, uint64_t allocation_from_log )
{
  m_cont_op = new ReplayOperation;
  auto alloc_op = m_alloc_operations[allocation_from_log];

  m_cont_op->op = [=]() {
    this->m_allocator_array[allocator_num].deallocate(alloc_op->m_allocation_ptr);
  };

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeCoalesce( const std::string allocator_name )
{
  m_cont_op = new ReplayOperation;

  m_cont_op->op = [=]() {
    auto& rm = umpire::ResourceManager::getInstance();
    auto alloc = rm.getAllocator(allocator_name);

    try {
      auto dynamic_pool =
        umpire::util::unwrap_allocator<umpire::strategy::DynamicPoolMap>(alloc);
      dynamic_pool->coalesce();
    }
    catch(...) {
      auto dynamic_pool =
        umpire::util::unwrap_allocator<umpire::strategy::DynamicPoolList>(alloc);
      dynamic_pool->coalesce();
    }
  };

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeRelease( int allocator_num )
{
  m_cont_op = new ReplayOperation;

  m_cont_op->op = [=]() {
    this->m_allocator_array[allocator_num].release();
  };

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAllocationMapInsert(void* key, umpire::util::AllocationRecord rec)
{
  m_cont_op = new ReplayOperation;

  m_cont_op->op = [=]() {
    this->m_allocation_map.insert(key, rec);
  };

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAllocationMapFind(void* key)
{
  m_cont_op = new ReplayOperation;
  m_cont_op->op = [=]() {
    this->m_allocation_map.find(key);
  };
  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAllocationMapRemove(void* key)
{
  m_cont_op = new ReplayOperation;

  m_cont_op->op = [=]() {
    this->m_allocation_map.remove(key);
  };

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAllocationMapClear(void)
{
  m_cont_op = new ReplayOperation;

  m_cont_op->op = [=]() {
    this->m_allocation_map.clear();
  };

  operations.push_back(m_cont_op);
}

