//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_ReplayOperationManager_HPP
#define REPLAY_ReplayOperationManager_HPP

#include <iostream>
#include <cstdint>
#include <vector>

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

class ReplayOperationManager;
struct ReplayOperation {
  std::function<void ()> op;
  void* m_allocation_ptr;
};

class ReplayOperationManager {
  using AllocationOpMap = std::unordered_map<uint64_t, ReplayOperation*>;

public:
  ReplayOperationManager( void );

  ~ReplayOperationManager();

  void runOperations(bool gather_statistics);

  void dumpStats();

  void makeMemoryResource( const std::string resource_name );

  template<typename Strategy, bool Introspection, typename... Args>
  void makeAllocator(
      const std::string allocator_name
    , const std::string base_allocator_name
    , Args&&... args);

  template<typename... Args>
  void makeFixedPool(
      const bool introspection
    , const std::string allocator_name
    , const std::string base_allocator_name
    , Args&&... args
  );

  template <typename... Args>
  void makeAdvisor(
      const bool introspection,
      const std::string allocator_name,
      const std::string base_allocator_name,
      Args&&... args
  );

  template <typename... Args>
  void makeAdvisor(
      const bool introspection,
      const std::string allocator_name,
      const std::string base_allocator_name,
      const std::string advice_operation,
      const std::string accessing_allocator_name,
      Args&&... args
  );

  template <typename... Args>
  void makeDynamicPoolMap(
      const bool introspection
    , const std::string allocator_name
    , const std::string base_allocator_name
    , Args&&... args
  );

  template <typename... Args>
  void makeDynamicPoolList(
      const bool introspection
    , const std::string allocator_name
    , const std::string base_allocator_name
    , Args&&... args
  );

  void makeMonotonicAllocator(
      const bool introspection
    , const std::string allocator_name
    , const std::size_t capacity
    , const std::string base_allocator_name
  );
  
  void makeSlotPool(
      const bool introspection
    , const std::string allocator_name
    , const std::size_t slots
    , const std::string base_allocator_name
  );

  void makeSizeLimiter(
      const bool introspection
    , const std::string allocator_name
    , const std::string base_allocator_name
    , const std::size_t size_limit
  );

  void makeThreadSafeAllocator(
      const bool introspection
    , const std::string allocator_name
    , const std::string base_allocator_name
  );

  template <typename... Args>
  void makeMixedPool(
      const bool introspection
    , const std::string allocator_name
    , const std::string base_allocator_name
    , Args&&... args
  );

  void makeAllocatorCont( void );

  //
  // Allocate/Deallocate
  //
  void makeAllocate( int allocator_num, std::size_t size );
  void makeAllocateCont( uint64_t allocation_from_log );
  void makeDeallocate( int allocator_num, uint64_t allocation_from_log );
  void makeCoalesce( const std::string allocator_name );
  void makeRelease( int allocator_num );

  void makeAllocationMapInsert(void* key, umpire::util::AllocationRecord rec);
  void makeAllocationMapFind(void* key);
  void makeAllocationMapRemove(void* key);
  void makeAllocationMapClear(void);

private:
  std::vector<umpire::Allocator> m_allocator_array;
  AllocationOpMap m_alloc_operations;
  ReplayOperation* m_cont_op;
  std::vector<ReplayOperation*> operations;
  std::vector<std::string> m_resource_names;
  umpire::util::AllocationMap m_allocation_map;

  std::map<std::string, std::vector< std::pair<size_t, std::size_t>>> m_stat_series;
};

#include "ReplayOperationManager.inl"

#endif // REPLAY_ReplayOperationManager_HPP
