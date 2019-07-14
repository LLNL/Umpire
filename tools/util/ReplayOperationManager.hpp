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
#include "umpire/ResourceManager.hpp"

class ReplayOperation {
public:
  using AllocationOpMap = std::unordered_map<uint64_t, ReplayOperation*>;
  std::function<void ()> op;

  ReplayOperation(
      std::vector<umpire::Allocator>& alloc_array,
      AllocationOpMap& alloc_operations
  );

  void runOperations();

  template <typename... Args>
  void makeAdvisor(
      const bool introspection,
      const std::string& allocator_name,
      const std::string& base_allocator_name,
      Args&&... args
  );

  template <typename... Args>
  void makeAdvisor(
      const bool introspection,
      const std::string& allocator_name,
      const std::string& base_allocator_name,
      const std::string& advice_operation,
      const std::string& accessing_allocator_name,
      Args&&... args
  );

  //
  // FixedPool
  //
  template<typename... Args>
  void makeFixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , Args&&... args
  );

  //
  // DynamicPool
  //
  void makeDynamicPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t initial_alloc_size
    , const std::size_t min_alloc_size
    , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
    , int alignment
  );

  void makeDynamicPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t initial_alloc_size
    , const std::size_t min_alloc_size
    , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
  );

  template <typename... Args>
  void makeDynamicPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , Args&&... args
  );

  void makeMonotonicAllocator(
      const bool introspection
    , const std::string& allocator_name
    , const std::size_t capacity
    , const std::string& base_allocator_name
  );

  void makeSlotPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::size_t slots
    , const std::string& base_allocator_name
  );

  void makeSizeLimiter(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t size_limit
  );

  void makeThreadSafeAllocator(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
  );

  void makeMixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t smallest_fixed_blocksize
    , const std::size_t largest_fixed_blocksize
    , const std::size_t max_fixed_blocksize
    , const std::size_t size_multiplier
    , const std::size_t dynamic_initial_alloc_bytes
    , const std::size_t dynamic_min_alloc_bytes
    , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
    , int alignment
  );

  void makeMixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t smallest_fixed_blocksize
    , const std::size_t largest_fixed_blocksize
    , const std::size_t max_fixed_blocksize
    , const std::size_t size_multiplier
    , const std::size_t dynamic_initial_alloc_bytes
    , const std::size_t dynamic_min_alloc_bytes
    , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
  );

  template <typename... Args>
  void makeMixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , Args&&... args
  );

  void makeAllocatorCont( void );
  void makeAllocate( int allocator_num, std::size_t size );
  void makeAllocateCont( uint64_t allocation_from_log );
  void makeDeallocate( int allocator_num, uint64_t allocation_from_log );
  void makeCoalesce(const std::string& allocator_name);
  void makeRelease(int allocator_num);

private:
  std::vector<umpire::Allocator>& m_alloc_array;
  AllocationOpMap& m_alloc_operations;
  void* m_allocation_ptr;
};

class ReplayOperationManager {
public:
  ReplayOperationManager( void );

  ~ReplayOperationManager();

  void runOperations();

  //
  // AllocationAdvisor
  //
  void makeAdvisor(
      const bool introspection,
      const std::string& allocator_name,
      const std::string& base_allocator_name,
      const std::string& advice_operation,
      const int device_id
  );

  void makeAdvisor(
      const bool introspection,
      const std::string& allocator_name,
      const std::string& base_allocator_name,
      const std::string& advice_operation,
      const std::string& accessing_allocator_name,
      const int device_id
  );

  void makeAdvisor(
      const bool introspection,
      const std::string& allocator_name,
      const std::string& base_allocator_name,
      const std::string& advice_operation
  );

  void makeAdvisor(
      const bool introspection,
      const std::string& allocator_name,
      const std::string& base_allocator_name,
      const std::string& advice_operation,
      const std::string& accessing_allocator_name
  );

  //
  // FixedPool
  //
  void makeFixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t object_bytes
    , const std::size_t objects_per_pool
  );
  
  void makeFixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t object_bytes
  );

  //
  // Dynamic Pool
  //
  void makeDynamicPool(
        const bool introspection
      , const std::string& allocator_name
      , const std::string& base_allocator_name
      , const std::size_t initial_alloc_size
      , const std::size_t min_alloc_size
      , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
      , int alignment
  );
  
  void makeDynamicPool(
        const bool introspection
      , const std::string& allocator_name
      , const std::string& base_allocator_name
      , const std::size_t initial_alloc_size
      , const std::size_t min_alloc_size
      , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
  );
  
  void makeDynamicPool(
        const bool introspection
      , const std::string& allocator_name
      , const std::string& base_allocator_name
      , const std::size_t initial_alloc_size
  );
 
  void makeDynamicPool(
        const bool introspection
      , const std::string& allocator_name
      , const std::string& base_allocator_name
  );

  void makeMonotonicAllocator(
      const bool introspection
    , const std::string& allocator_name
    , const std::size_t capacity
    , const std::string& base_allocator_name
  );
  
  void makeSlotPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::size_t slots
    , const std::string& base_allocator_name
  );

  void makeSizeLimiter(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t size_limit
  );

  void makeThreadSafeAllocator(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
  );

  void makeMixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t smallest_fixed_blocksize
    , const std::size_t largest_fixed_blocksize
    , const std::size_t max_fixed_blocksize
    , const std::size_t size_multiplier
    , const std::size_t dynamic_initial_alloc_bytes
    , const std::size_t dynamic_min_alloc_bytes
    , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
    , int alignment
  );

  void makeMixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t smallest_fixed_blocksize
    , const std::size_t largest_fixed_blocksize
    , const std::size_t max_fixed_blocksize
    , const std::size_t size_multiplier
    , const std::size_t dynamic_initial_alloc_bytes
    , const std::size_t dynamic_min_alloc_bytes
    , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
  );

  void makeMixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t smallest_fixed_blocksize
    , const std::size_t largest_fixed_blocksize
    , const std::size_t max_fixed_blocksize
    , const std::size_t size_multiplier
    , const std::size_t dynamic_initial_alloc_bytes
  );

  void makeMixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t smallest_fixed_blocksize
    , const std::size_t largest_fixed_blocksize
    , const std::size_t max_fixed_blocksize
    , const std::size_t size_multiplier
  );
  
  void makeMixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t smallest_fixed_blocksize
    , const std::size_t largest_fixed_blocksize
    , const std::size_t max_fixed_blocksize
  );

  void makeMixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t smallest_fixed_blocksize
    , const std::size_t largest_fixed_blocksize
  );

  void makeMixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t smallest_fixed_blocksize
  );

  void makeMixedPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
  );

  void makeAllocatorCont( void );

  //
  // Allocate/Deallocate
  //
  void makeAllocate( int allocator_num, std::size_t size );
  void makeAllocateCont( uint64_t allocation_from_log );
  void makeDeallocate( int allocator_num, uint64_t allocation_from_log );
  void makeCoalesce( const std::string& allocator_name );
  void makeRelease( int allocator_num );

private:
  std::vector<umpire::Allocator> m_allocator_array;
  ReplayOperation::AllocationOpMap m_alloc_operations;
  ReplayOperation* m_cont_op;
  std::vector<ReplayOperation*> operations;
};

#include "util/ReplayOperationManager.inl"

#endif // REPLAY_ReplayOperationManager_HPP
