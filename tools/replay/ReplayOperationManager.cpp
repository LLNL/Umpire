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

void ReplayOperationManager::makeAllocate( int allocator_num, std::size_t size )
{
  auto alloc_op = new ReplayOperation;
  alloc_op->op = [=]() {
    alloc_op->m_allocation_ptr = this->m_allocator_array[allocator_num].allocate(size);
  };
  m_cont_op = alloc_op;
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

