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
      std::vector<umpire::Allocator*>& alloc_array,
      AllocationOpMap& alloc_operations
  ) :   m_alloc_array(alloc_array)
      , m_alloc_operations(alloc_operations)
  {
  }

  void run()
  {
    op();
  }

  //
  // AllocationAdvisor
  //
  template <typename... Args>
  void bld_advisor(
      const bool introspection,
      const std::string& allocator_name,
      const std::string& base_allocator_name,
      Args&&... args
  )
  {
    if (introspection) {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::AllocationAdvisor, true>
          ( allocator_name, rm.getAllocator(base_allocator_name),
            std::forward<Args>(args)...);

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
    else {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::AllocationAdvisor, false>
          ( allocator_name, rm.getAllocator(base_allocator_name),
            std::forward<Args>(args)...);

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
  }

  template <typename... Args>
  void bld_advisor(
      const bool introspection,
      const std::string& allocator_name,
      const std::string& base_allocator_name,
      const std::string& advice_operation,
      const std::string& accessing_allocator_name,
      Args&&... args
  )
  {
    if (introspection) {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::AllocationAdvisor, true>
          ( allocator_name, rm.getAllocator(base_allocator_name),
            advice_operation, rm.getAllocator(accessing_allocator_name), 
            std::forward<Args>(args)...);

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
    else {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::AllocationAdvisor, false>
          ( allocator_name, rm.getAllocator(base_allocator_name),
            advice_operation, rm.getAllocator(accessing_allocator_name),
            std::forward<Args>(args)...);

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
  }

  //
  // FixedPool
  //
  template<typename... Args>
  void bld_fixedpool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , Args&&... args
  )
  {
    if (introspection) {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::FixedPool, true>
          (allocator_name, rm.getAllocator(base_allocator_name), std::forward<Args>(args)...);

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
    else {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::FixedPool, false>
          (allocator_name, rm.getAllocator(base_allocator_name), std::forward<Args>(args)...);

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
  }

  //
  // DynamicPool
  //
  void bld_dynamicpool(
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

        rm.makeAllocator<umpire::strategy::DynamicPool, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , initial_alloc_size
            , min_alloc_size
            , umpire::strategy::heuristic_percent_releasable(0)
            , alignment
          );

        auto allocator = new umpire::Allocator(
                                rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
    else {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::DynamicPool, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , initial_alloc_size
            , min_alloc_size
            , umpire::strategy::heuristic_percent_releasable(0)
            , alignment
          );

        auto allocator = new umpire::Allocator(
                                rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
  }

  void bld_dynamicpool(
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

        rm.makeAllocator<umpire::strategy::DynamicPool, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , initial_alloc_size
            , min_alloc_size
            , umpire::strategy::heuristic_percent_releasable(0)
          );

        auto allocator = new umpire::Allocator(
                                rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
    else {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::DynamicPool, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , initial_alloc_size
            , min_alloc_size
            , umpire::strategy::heuristic_percent_releasable(0)
          );

        auto allocator = new umpire::Allocator(
                                rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
  }
  template <typename... Args>
  void bld_dynamicpool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , Args&&... args
  )
  {
    if (introspection) {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::DynamicPool, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args)...
          );

        auto allocator = new umpire::Allocator(
                                rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
    else {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::DynamicPool, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args)...
          );

        auto allocator = new umpire::Allocator(
                                rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
  }

  void bld_monotonic(
      const bool introspection
    , const std::string& allocator_name
    , const std::size_t capacity
    , const std::string& base_allocator_name
  )
  {
    if (introspection) {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy, true>
          (   allocator_name
            , capacity
            , rm.getAllocator(base_allocator_name)
          );

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
    else {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy, false>
          (   allocator_name
            , capacity
            , rm.getAllocator(base_allocator_name)
          );

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
  }

  void bld_slotpool(
      const bool introspection
    , const std::string& allocator_name
    , const std::size_t slots
    , const std::string& base_allocator_name
  )
  {
    if (introspection) {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::SlotPool, true>
          (   allocator_name
            , slots
            , rm.getAllocator(base_allocator_name)
          );

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
    else {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::SlotPool, false>
          (   allocator_name
            , slots
            , rm.getAllocator(base_allocator_name)
          );

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
  }

  void bld_limiter(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , const std::size_t size_limit
  )
  {
    if (introspection) {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::SizeLimiter, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , size_limit
          );

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
    else {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::SizeLimiter, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , size_limit
          );

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
  }

  void bld_threadsafe(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
  )
  {
    if (introspection) {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
          );

        auto allocator = new umpire::Allocator(
                                rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
    else {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
          );

        auto allocator = new umpire::Allocator(
                                rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
  }

  void bld_mixedpool(
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

        rm.makeAllocator<umpire::strategy::MixedPool, true>
          (   
              allocator_name
            , rm.getAllocator(base_allocator_name)
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
            , max_fixed_blocksize
            , size_multiplier
            , dynamic_initial_alloc_bytes
            , dynamic_min_alloc_bytes
            , umpire::strategy::heuristic_percent_releasable(0)
            , alignment
          );

        auto allocator = new umpire::Allocator(
                                rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
    else {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::MixedPool, false>
          (   
              allocator_name
            , rm.getAllocator(base_allocator_name)
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
            , max_fixed_blocksize
            , size_multiplier
            , dynamic_initial_alloc_bytes
            , dynamic_min_alloc_bytes
            , umpire::strategy::heuristic_percent_releasable(0)
            , alignment
          );

        auto allocator = new umpire::Allocator(
                                rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
  }

  void bld_mixedpool(
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

        rm.makeAllocator<umpire::strategy::MixedPool, true>
          (   
              allocator_name
            , rm.getAllocator(base_allocator_name)
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
            , max_fixed_blocksize
            , size_multiplier
            , dynamic_initial_alloc_bytes
            , dynamic_min_alloc_bytes
            , umpire::strategy::heuristic_percent_releasable(0)
          );

        auto allocator = new umpire::Allocator(
                                rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
    else {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::MixedPool, false>
          (   
              allocator_name
            , rm.getAllocator(base_allocator_name)
            , smallest_fixed_blocksize
            , largest_fixed_blocksize
            , max_fixed_blocksize
            , size_multiplier
            , dynamic_initial_alloc_bytes
            , dynamic_min_alloc_bytes
            , umpire::strategy::heuristic_percent_releasable(0)
          );

        auto allocator = new umpire::Allocator(
                                rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
  }

  template <typename... Args>
  void bld_mixedpool(
      const bool introspection
    , const std::string& allocator_name
    , const std::string& base_allocator_name
    , Args&&... args
  )
  {
    if (introspection) {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::MixedPool, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args)...
          );

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
    else {
      op = [=]() {
        auto& rm = umpire::ResourceManager::getInstance();

        rm.makeAllocator<umpire::strategy::MixedPool, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args)...
          );

        auto allocator = new umpire::Allocator(rm.getAllocator(allocator_name));
        this->m_alloc_array.push_back(allocator);
      };
    }
  }

  void bld_allocator_cont( void )
  {
  }

  void bld_allocate( int allocator_num, std::size_t size )
  {
    op = [=]() {
      this->m_allocation_ptr = this->m_alloc_array[allocator_num]->allocate(size);
    };
  }

  void bld_allocate_cont( uint64_t allocation_from_log )
  {
    m_alloc_operations[allocation_from_log] = this;
  }

  void bld_deallocate( int allocator_num, uint64_t allocation_from_log )
  {
    auto alloc_op = m_alloc_operations[allocation_from_log];

    op = [=]() {
      this->m_alloc_array[allocator_num]->deallocate(alloc_op->m_allocation_ptr);
    };
  }

  void bld_coalesce(
    const std::string& allocator_name
  )
  {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();
      auto alloc = rm.getAllocator(allocator_name);
      auto strategy = alloc.getAllocationStrategy();
      auto tracker = dynamic_cast<umpire::strategy::AllocationTracker*>(strategy);

      if (tracker)
        strategy = tracker->getAllocationStrategy();

      auto dynamic_pool = dynamic_cast<umpire::strategy::DynamicPool*>(strategy);

      if (dynamic_pool)
        dynamic_pool->coalesce();
      else
        std::cerr << allocator_name << " is not a dynamic pool, skipping\n";
    };
  }

  void bld_release( int allocator_num )
  {
    op = [=]() {
      this->m_alloc_array[allocator_num]->release();
    };
  }

private:
  std::vector<umpire::Allocator*>& m_alloc_array;
  AllocationOpMap& m_alloc_operations;
  void* m_allocation_ptr;
};

class ReplayOperationManager {
  public:
    ReplayOperationManager( void ) {
    }

    ~ReplayOperationManager() { }

    void run() {
      for (auto op : operations) {
        op->run();
      }
    }

    //
    // AllocationAdvisor
    //
    void bld_advisor(
        const bool introspection,
        const std::string& allocator_name,
        const std::string& base_allocator_name,
        const std::string& advice_operation,
        const int device_id
    ) {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_advisor(
          introspection, allocator_name, base_allocator_name,
          advice_operation, device_id);
      operations.push_back(m_cont_op);
    }

    void bld_advisor(
        const bool introspection,
        const std::string& allocator_name,
        const std::string& base_allocator_name,
        const std::string& advice_operation,
        const std::string& accessing_allocator_name,
        const int device_id
    ) {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_advisor(
          introspection, allocator_name, base_allocator_name,
          advice_operation, accessing_allocator_name, device_id);
      operations.push_back(m_cont_op);
    }

    void bld_advisor(
        const bool introspection,
        const std::string& allocator_name,
        const std::string& base_allocator_name,
        const std::string& advice_operation
    ) {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_advisor(
          introspection, allocator_name, base_allocator_name,
          advice_operation);
      operations.push_back(m_cont_op);
    }

    void bld_advisor(
        const bool introspection,
        const std::string& allocator_name,
        const std::string& base_allocator_name,
        const std::string& advice_operation,
        const std::string& accessing_allocator_name
    ) {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_advisor(
          introspection, allocator_name, base_allocator_name,
          advice_operation, accessing_allocator_name);
      operations.push_back(m_cont_op);
    }

    //
    // FixedPool
    //
    void bld_fixedpool(
        const bool introspection
      , const std::string& allocator_name
      , const std::string& base_allocator_name
      , const std::size_t object_bytes
      , const std::size_t objects_per_pool
    )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_fixedpool(
                  introspection
                , allocator_name
                , base_allocator_name
                , object_bytes
                , objects_per_pool
      );

      operations.push_back(m_cont_op);
    }

    void bld_fixedpool(
        const bool introspection
      , const std::string& allocator_name
      , const std::string& base_allocator_name
      , const std::size_t object_bytes
    )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_fixedpool(
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
    void bld_dynamicpool(
          const bool introspection
        , const std::string& allocator_name
        , const std::string& base_allocator_name
        , const std::size_t initial_alloc_size
        , const std::size_t min_alloc_size
        , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
        , int alignment
    ) {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_dynamicpool(
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

    void bld_dynamicpool(
          const bool introspection
        , const std::string& allocator_name
        , const std::string& base_allocator_name
        , const std::size_t initial_alloc_size
        , const std::size_t min_alloc_size
        , umpire::strategy::DynamicPool::CoalesceHeuristic /* h_fun */
    ) {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_dynamicpool(
                  introspection
                , allocator_name
                , base_allocator_name
                , initial_alloc_size
                , min_alloc_size
                , umpire::strategy::heuristic_percent_releasable(0)
      );

      operations.push_back(m_cont_op);
    }

    void bld_dynamicpool(
          const bool introspection
        , const std::string& allocator_name
        , const std::string& base_allocator_name
        , const std::size_t initial_alloc_size
    ) {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_dynamicpool(
                  introspection
                , allocator_name
                , base_allocator_name
                , initial_alloc_size
      );

      operations.push_back(m_cont_op);
    }

    void bld_dynamicpool(
          const bool introspection
        , const std::string& allocator_name
        , const std::string& base_allocator_name
    ) {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_dynamicpool(
                  introspection
                , allocator_name
                , base_allocator_name
      );

      operations.push_back(m_cont_op);
    }

    void bld_monotonic(
        const bool introspection
      , const std::string& allocator_name
      , const std::size_t capacity
      , const std::string& base_allocator_name
    )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_monotonic(
                  introspection
                , allocator_name
                , capacity
                , base_allocator_name
      );

      operations.push_back(m_cont_op);
    }

    void bld_slotpool(
        const bool introspection
      , const std::string& allocator_name
      , const std::size_t slots
      , const std::string& base_allocator_name
    )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_slotpool(
                  introspection
                , allocator_name
                , slots
                , base_allocator_name
      );

      operations.push_back(m_cont_op);
    }

    void bld_limiter(
        const bool introspection
      , const std::string& allocator_name
      , const std::string& base_allocator_name
      , const std::size_t size_limit
    )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_limiter(
                  introspection
                , allocator_name
                , base_allocator_name
                , size_limit
      );

      operations.push_back(m_cont_op);
    }

    void bld_threadsafe(
        const bool introspection
      , const std::string& allocator_name
      , const std::string& base_allocator_name
    )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_threadsafe(
                  introspection
                , allocator_name
                , base_allocator_name
      );

      operations.push_back(m_cont_op);
    }

    void bld_mixedpool(
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
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_mixedpool(
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

    void bld_mixedpool(
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
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_mixedpool(
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

    void bld_mixedpool(
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
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_mixedpool(
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

    void bld_mixedpool(
        const bool introspection
      , const std::string& allocator_name
      , const std::string& base_allocator_name
      , const std::size_t smallest_fixed_blocksize
      , const std::size_t largest_fixed_blocksize
      , const std::size_t max_fixed_blocksize
      , const std::size_t size_multiplier
    )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_mixedpool(
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

    void bld_mixedpool(
        const bool introspection
      , const std::string& allocator_name
      , const std::string& base_allocator_name
      , const std::size_t smallest_fixed_blocksize
      , const std::size_t largest_fixed_blocksize
      , const std::size_t max_fixed_blocksize
    )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_mixedpool(
                  introspection
                , allocator_name
                , base_allocator_name
                , smallest_fixed_blocksize
                , largest_fixed_blocksize
                , max_fixed_blocksize
      );

      operations.push_back(m_cont_op);
    }

    void bld_mixedpool(
        const bool introspection
      , const std::string& allocator_name
      , const std::string& base_allocator_name
      , const std::size_t smallest_fixed_blocksize
      , const std::size_t largest_fixed_blocksize
    )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_mixedpool(
                  introspection
                , allocator_name
                , base_allocator_name
                , smallest_fixed_blocksize
                , largest_fixed_blocksize
      );

      operations.push_back(m_cont_op);
    }

    void bld_mixedpool(
        const bool introspection
      , const std::string& allocator_name
      , const std::string& base_allocator_name
      , const std::size_t smallest_fixed_blocksize
    )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_mixedpool(
                  introspection
                , allocator_name
                , base_allocator_name
                , smallest_fixed_blocksize
      );

      operations.push_back(m_cont_op);
    }

    void bld_mixedpool(
        const bool introspection
      , const std::string& allocator_name
      , const std::string& base_allocator_name
    )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);

      m_cont_op->bld_mixedpool(
                  introspection
                , allocator_name
                , base_allocator_name
      );

      operations.push_back(m_cont_op);
    }

    void bld_allocator_cont( void )
    {
      m_cont_op->bld_allocator_cont();
    }

    //
    // Allocate/Deallocate
    //
    void bld_allocate( int allocator_num, std::size_t size )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);
      m_cont_op->bld_allocate(allocator_num, size);
      operations.push_back(m_cont_op);
    }

    void bld_allocate_cont( uint64_t allocation_from_log )
    {
      m_cont_op->bld_allocate_cont(allocation_from_log);
    }

    void bld_deallocate( int allocator_num, uint64_t allocation_from_log )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);
      m_cont_op->bld_deallocate(allocator_num, allocation_from_log);
      operations.push_back(m_cont_op);
    }

    void bld_coalesce( const std::string& allocator_name )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);
      m_cont_op->bld_coalesce(allocator_name);
      operations.push_back(m_cont_op);
    }

    void bld_release( int allocator_num )
    {
      m_cont_op = new ReplayOperation(m_allocator_array, m_alloc_operations);
      m_cont_op->bld_release(allocator_num);
      operations.push_back(m_cont_op);
    }

  private:
    std::vector<umpire::Allocator*> m_allocator_array;
    ReplayOperation::AllocationOpMap m_alloc_operations;
    ReplayOperation* m_cont_op;
    std::vector<ReplayOperation*> operations;
};

#endif // REPLAY_ReplayOperationManager_HPP
