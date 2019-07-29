//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
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
#include "util/ReplayMacros.hpp"
#include "util/ReplayOperationManager.hpp"

ReplayOperation::ReplayOperation(
    ReplayOperationManager& my_manager
) :   m_my_manager(my_manager)
{
}

void ReplayOperation::makeMemoryResources( void )
{
  auto& rm = umpire::ResourceManager::getInstance();

  for ( auto resource_name : m_my_manager.m_resource_names ) {
    m_my_manager.m_allocator_array.push_back(rm.getAllocator(resource_name));
  }
}

void ReplayOperation::runOperations()
{
  op();
}

//
// DynamicPool
//
void ReplayOperation::makeDynamicPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , const std::size_t initial_alloc_size
  , const std::size_t min_alloc_size
  , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
  , int alignment
)
{
  if (introspection) {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::DynamicPool, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , initial_alloc_size
            , min_alloc_size
            , umpire::strategy::heuristic_percent_releasable(0)
            , alignment
          )
      );
    };
  }
  else {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::DynamicPool, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , initial_alloc_size
            , min_alloc_size
            , umpire::strategy::heuristic_percent_releasable(0)
            , alignment
          )
      );
    };
  }
}

void ReplayOperation::makeDynamicPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , const std::size_t initial_alloc_size
  , const std::size_t min_alloc_size
  , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
)
{
  if (introspection) {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::DynamicPool, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , initial_alloc_size
            , min_alloc_size
            , umpire::strategy::heuristic_percent_releasable(0)
          )
      );
    };
  }
  else {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::DynamicPool, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , initial_alloc_size
            , min_alloc_size
            , umpire::strategy::heuristic_percent_releasable(0)
          )
      );
    };
  }
}

void ReplayOperation::makeMonotonicAllocator(
    const bool introspection
  , const std::string& allocator_name
  , const std::size_t capacity
  , const std::string& base_allocator_name
)
{
  if (introspection) {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy, true>
          (   allocator_name
            , capacity
            , rm.getAllocator(base_allocator_name)
          )
      );
    };
  }
  else {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy, false>
          (   allocator_name
            , capacity
            , rm.getAllocator(base_allocator_name)
          )
      );
    };
  }
}

void ReplayOperation::makeSlotPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::size_t slots
  , const std::string& base_allocator_name
)
{
  if (introspection) {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::SlotPool, true>
          (   allocator_name
            , slots
            , rm.getAllocator(base_allocator_name)
          )
      );
    };
  }
  else {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::SlotPool, false>
          (   allocator_name
            , slots
            , rm.getAllocator(base_allocator_name)
          )
      );
    };
  }
}

void ReplayOperation::makeSizeLimiter(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , const std::size_t size_limit
)
{
  if (introspection) {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::SizeLimiter, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , size_limit
          )
      );
    };
  }
  else {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::SizeLimiter, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , size_limit
          )
      );
    };
  }
}

void ReplayOperation::makeThreadSafeAllocator(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
)
{
  if (introspection) {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
          )
      );
    };
  }
  else {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
          )
      );
    };
  }
}

void ReplayOperation::makeMixedPool(
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
)
{
  if (introspection) {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::MixedPool, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
            , max_fixed_blocksize
            , size_multiplier
            , dynamic_initial_alloc_bytes
            , dynamic_min_alloc_bytes
            , umpire::strategy::heuristic_percent_releasable(0)
            , alignment
          )
      );
    };
  }
  else {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::MixedPool, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
            , max_fixed_blocksize
            , size_multiplier
            , dynamic_initial_alloc_bytes
            , dynamic_min_alloc_bytes
            , umpire::strategy::heuristic_percent_releasable(0)
            , alignment
          )
      );
    };
  }
}

void ReplayOperation::makeMixedPool(
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
)
{
  if (introspection) {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::MixedPool, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
            , max_fixed_blocksize
            , size_multiplier
            , dynamic_initial_alloc_bytes
            , dynamic_min_alloc_bytes
            , umpire::strategy::heuristic_percent_releasable(0)
          )
      );
    };
  }
  else {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::MixedPool, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
            , max_fixed_blocksize
            , size_multiplier
            , dynamic_initial_alloc_bytes
            , dynamic_min_alloc_bytes
            , umpire::strategy::heuristic_percent_releasable(0)
          )
      );
    };
  }
}

void ReplayOperation::makeAllocatorCont( void )
{
}

void ReplayOperation::makeAllocate( int allocator_num, std::size_t size )
{
  op = [=]() {
    this->m_allocation_ptr = this->m_my_manager.m_allocator_array[allocator_num].allocate(size);
  };
}

void ReplayOperation::makeAllocateCont( uint64_t allocation_from_log )
{
  m_my_manager.m_alloc_operations[allocation_from_log] = this;
}

void ReplayOperation::makeDeallocate( int allocator_num, uint64_t allocation_from_log )
{
  auto alloc_op = m_my_manager.m_alloc_operations[allocation_from_log];

  op = [=]() {
    this->m_my_manager.m_allocator_array[allocator_num].deallocate(alloc_op->m_allocation_ptr);
  };
}

void ReplayOperation::makeCoalesce(
  const std::string& allocator_name
)
{
  op = [=]() {
    auto& rm = umpire::ResourceManager::getInstance();
    auto alloc = rm.getAllocator(allocator_name);
    auto dynamic_pool =
      umpire::util::unwrap_allocator<umpire::strategy::DynamicPool>(alloc);

    if (dynamic_pool)
      dynamic_pool->coalesce();
    else
      REPLAY_ERROR(allocator_name << " is not a dynamic pool, skipping");
  };
}

void ReplayOperation::makeRelease( int allocator_num )
{
  op = [=]() {
    this->m_my_manager.m_allocator_array[allocator_num].release();
  };
}

void ReplayOperation::makeAllocationMapInsert(void* key, umpire::util::AllocationRecord rec)
{
  op = [=]() {
    this->m_my_manager.m_allocation_map.insert(key, rec);
  };
}

void ReplayOperation::makeAllocationMapFind(void* key)
{
  op = [=]() {
    this->m_my_manager.m_allocation_map.find(key);
  };
}

void ReplayOperation::makeAllocationMapRemove(void* key)
{
  op = [=]() {
    this->m_my_manager.m_allocation_map.remove(key);
  };
}

void ReplayOperation::makeAllocationMapClear(void)
{
  op = [=]() {
    this->m_my_manager.m_allocation_map.clear();
  };
}

ReplayOperationManager::ReplayOperationManager( void ) {
}

ReplayOperationManager::~ReplayOperationManager() { }

void ReplayOperationManager::runOperations()
{
  ReplayOperation m_op(*this);

  m_op.makeMemoryResources();

  for (auto op : operations) {
    op->runOperations();
  }
}

void ReplayOperationManager::makeMemoryResource( const std::string& resource_name )
{
  m_resource_names.push_back(resource_name);
}

//
// AllocationAdvisor
//
void ReplayOperationManager::makeAdvisor(
    const bool introspection,
    const std::string& allocator_name,
    const std::string& base_allocator_name,
    const std::string& advice_operation,
    const int device_id
) {
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeAdvisor(
      introspection, allocator_name, base_allocator_name,
      advice_operation, device_id);
  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAdvisor(
    const bool introspection,
    const std::string& allocator_name,
    const std::string& base_allocator_name,
    const std::string& advice_operation,
    const std::string& accessing_allocator_name,
    const int device_id
) {
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeAdvisor(
      introspection, allocator_name, base_allocator_name,
      advice_operation, accessing_allocator_name, device_id);
  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAdvisor(
    const bool introspection,
    const std::string& allocator_name,
    const std::string& base_allocator_name,
    const std::string& advice_operation
) {
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeAdvisor(
      introspection, allocator_name, base_allocator_name,
      advice_operation);
  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAdvisor(
    const bool introspection,
    const std::string& allocator_name,
    const std::string& base_allocator_name,
    const std::string& advice_operation,
    const std::string& accessing_allocator_name
) {
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeAdvisor(
      introspection, allocator_name, base_allocator_name,
      advice_operation, accessing_allocator_name);
  operations.push_back(m_cont_op);
}

//
// FixedPool
//
void ReplayOperationManager::makeFixedPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , const std::size_t object_bytes
  , const std::size_t objects_per_pool
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeFixedPool(
              introspection
            , allocator_name
            , base_allocator_name
            , object_bytes
            , objects_per_pool
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeFixedPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , const std::size_t object_bytes
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeFixedPool(
              introspection
            , allocator_name
            , base_allocator_name
            , object_bytes
  );

  operations.push_back(m_cont_op);
}

//
// Dynamic Pool
//
void ReplayOperationManager::makeDynamicPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t initial_alloc_size
    , const std::size_t min_alloc_size
    , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
    , int alignment
) {
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeDynamicPool(
              introspection
            , allocator_name
            , base_allocator_name
            , initial_alloc_size
            , min_alloc_size
            , umpire::strategy::heuristic_percent_releasable(0)
            , alignment
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeDynamicPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t initial_alloc_size
    , const std::size_t min_alloc_size
    , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
) {
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeDynamicPool(
              introspection
            , allocator_name
            , base_allocator_name
            , initial_alloc_size
            , min_alloc_size
            , umpire::strategy::heuristic_percent_releasable(0)
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeDynamicPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t initial_alloc_size
) {
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeDynamicPool(
              introspection
            , allocator_name
            , base_allocator_name
            , initial_alloc_size
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeDynamicPool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
) {
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeDynamicPool(
              introspection
            , allocator_name
            , base_allocator_name
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeMonotonicAllocator(
    const bool introspection
  , const std::string& allocator_name
  , const std::size_t capacity
  , const std::string& base_allocator_name
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeMonotonicAllocator(
              introspection
            , allocator_name
            , capacity
            , base_allocator_name
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeSlotPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::size_t slots
  , const std::string& base_allocator_name
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeSlotPool(
              introspection
            , allocator_name
            , slots
            , base_allocator_name
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeSizeLimiter(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , const std::size_t size_limit
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeSizeLimiter(
              introspection
            , allocator_name
            , base_allocator_name
            , size_limit
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeThreadSafeAllocator(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeThreadSafeAllocator(
              introspection
            , allocator_name
            , base_allocator_name
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeMixedPool(
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
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeMixedPool(
              introspection
            , allocator_name
            , base_allocator_name
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
            , max_fixed_blocksize
            , size_multiplier
            , dynamic_initial_alloc_bytes
            , dynamic_min_alloc_bytes
            , umpire::strategy::heuristic_percent_releasable(0)
            , alignment
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeMixedPool(
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
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeMixedPool(
              introspection
            , allocator_name
            , base_allocator_name
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
            , max_fixed_blocksize
            , size_multiplier
            , dynamic_initial_alloc_bytes
            , dynamic_min_alloc_bytes
            , umpire::strategy::heuristic_percent_releasable(0)
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeMixedPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , const std::size_t smallest_fixed_blocksize
  , const std::size_t largest_fixed_blocksize
  , const std::size_t max_fixed_blocksize
  , const std::size_t size_multiplier
  , const std::size_t dynamic_initial_alloc_bytes
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeMixedPool(
              introspection
            , allocator_name
            , base_allocator_name
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
            , max_fixed_blocksize
            , size_multiplier
            , dynamic_initial_alloc_bytes
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeMixedPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , const std::size_t smallest_fixed_blocksize
  , const std::size_t largest_fixed_blocksize
  , const std::size_t max_fixed_blocksize
  , const std::size_t size_multiplier
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeMixedPool(
              introspection
            , allocator_name
            , base_allocator_name
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
            , max_fixed_blocksize
            , size_multiplier
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeMixedPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , const std::size_t smallest_fixed_blocksize
  , const std::size_t largest_fixed_blocksize
  , const std::size_t max_fixed_blocksize
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeMixedPool(
              introspection
            , allocator_name
            , base_allocator_name
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
            , max_fixed_blocksize
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeMixedPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , const std::size_t smallest_fixed_blocksize
  , const std::size_t largest_fixed_blocksize
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeMixedPool(
              introspection
            , allocator_name
            , base_allocator_name
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeMixedPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , const std::size_t smallest_fixed_blocksize
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeMixedPool(
              introspection
            , allocator_name
            , base_allocator_name
            , smallest_fixed_blocksize
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeMixedPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
)
{
  m_cont_op = new ReplayOperation(*this);

  m_cont_op->makeMixedPool(
              introspection
            , allocator_name
            , base_allocator_name
  );

  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAllocatorCont( void )
{
  m_cont_op->makeAllocatorCont();
}

//
// Allocate/Deallocate
//
void ReplayOperationManager::makeAllocate( int allocator_num, std::size_t size )
{
  m_cont_op = new ReplayOperation(*this);
  m_cont_op->makeAllocate(allocator_num, size);
  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAllocateCont( uint64_t allocation_from_log )
{
  m_cont_op->makeAllocateCont(allocation_from_log);
}

void ReplayOperationManager::makeDeallocate( int allocator_num, uint64_t allocation_from_log )
{
  m_cont_op = new ReplayOperation(*this);
  m_cont_op->makeDeallocate(allocator_num, allocation_from_log);
  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeCoalesce( const std::string& allocator_name )
{
  m_cont_op = new ReplayOperation(*this);
  m_cont_op->makeCoalesce(allocator_name);
  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeRelease( int allocator_num )
{
  m_cont_op = new ReplayOperation(*this);
  m_cont_op->makeRelease(allocator_num);
  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAllocationMapInsert(void* key, umpire::util::AllocationRecord rec)
{
  m_cont_op = new ReplayOperation(*this);
  m_cont_op->makeAllocationMapInsert(key, rec);
  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAllocationMapFind(void* key)
{
  m_cont_op = new ReplayOperation(*this);
  m_cont_op->makeAllocationMapFind(key);
  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAllocationMapRemove(void* key)
{
  m_cont_op = new ReplayOperation(*this);
  m_cont_op->makeAllocationMapRemove(key);
  operations.push_back(m_cont_op);
}

void ReplayOperationManager::makeAllocationMapClear(void)
{
  m_cont_op = new ReplayOperation(*this);
  m_cont_op->makeAllocationMapClear();
  operations.push_back(m_cont_op);
}

