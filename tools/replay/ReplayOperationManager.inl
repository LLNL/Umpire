//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_ReplayOperationManager_INL
#define REPLAY_ReplayOperationManager_INL

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

template<typename Strategy, bool Introspection, typename... Args>
void 
ReplayOperationManager::makeAllocator(
    const std::string allocator_name
  , const std::string base_allocator_name
  , Args&&... args)
{
  m_cont_op = new ReplayOperation;
  m_cont_op->op = [=]() {
    auto& rm = umpire::ResourceManager::getInstance();
    this->m_allocator_array.push_back(
        rm.makeAllocator<Strategy, Introspection>(
          allocator_name,
          rm.getAllocator(base_allocator_name),
          args...));
  };
  operations.push_back(m_cont_op);
}


//
// AllocationAdvisor
//
template <typename... Args>
void ReplayOperationManager::makeAdvisor(
    const bool introspection,
    const std::string allocator_name,
    const std::string base_allocator_name,
    Args&&... args
)
{
  m_cont_op = new ReplayOperation;

  if (introspection) {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::AllocationAdvisor, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args) ...
          )
      );
    };
  }
  else {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::AllocationAdvisor, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args) ...
          )
      );
    };
  }

  operations.push_back(m_cont_op);
}

template <typename... Args>
void ReplayOperationManager::makeAdvisor(
    const bool introspection,
    const std::string allocator_name,
    const std::string base_allocator_name,
    const std::string advice_operation,
    const std::string accessing_allocator_name,
    Args&&... args
)
{
  m_cont_op = new ReplayOperation;

  if (introspection) {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::AllocationAdvisor, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , advice_operation
            , rm.getAllocator(accessing_allocator_name)
            , std::forward<Args>(args)...
          )
      );
    };
  }
  else {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::AllocationAdvisor, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , advice_operation
            , rm.getAllocator(accessing_allocator_name)
            , std::forward<Args>(args)...
          )
      );
    };
  }

  operations.push_back(m_cont_op);
}

template <typename... Args>
void ReplayOperationManager::makeDynamicPoolMap(
    const bool introspection
  , const std::string allocator_name
  , const std::string base_allocator_name
  , Args&&... args
)
{
  m_cont_op = new ReplayOperation;

  if (introspection) {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::DynamicPoolMap, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args)...
          )
      );
    };
  }
  else {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::DynamicPoolMap, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args)...
          )
      );
    };
  }

  operations.push_back(m_cont_op);
}

template <typename... Args>
void ReplayOperationManager::makeDynamicPoolList(
    const bool introspection
  , const std::string allocator_name
  , const std::string base_allocator_name
  , Args&&... args
)
{
  m_cont_op = new ReplayOperation;

  if (introspection) {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::DynamicPoolList, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args)...
          )
      );
    };
  }
  else {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::DynamicPoolList, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args)...
          )
      );
    };
  }

  operations.push_back(m_cont_op);
}

//
// FixedPool
//
template<typename... Args>
void ReplayOperationManager::makeFixedPool(
    const bool introspection
  , const std::string allocator_name
  , const std::string base_allocator_name
  , Args&&... args
)
{
  m_cont_op = new ReplayOperation;

  if (introspection) {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::FixedPool, true>
          (  allocator_name
           , rm.getAllocator(base_allocator_name)
           , std::forward<Args>(args)...
          )
      );
    };
  }
  else {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::FixedPool, false>
          (  allocator_name
           , rm.getAllocator(base_allocator_name)
           , std::forward<Args>(args)...
          )
      );
    };
  }

  operations.push_back(m_cont_op);
}

template <typename... Args>
void ReplayOperationManager::makeMixedPool(
    const bool introspection
  , const std::string allocator_name
  , const std::string base_allocator_name
  , Args&&... args
)
{
  m_cont_op = new ReplayOperation;

  if (introspection) {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::MixedPool, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , (args)...
          )
      );
    };
  }
  else {
    m_cont_op->op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::MixedPool, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , (args)...
          )
      );
    };
  }

  operations.push_back(m_cont_op);
}
#endif // REPLAY_ReplayOperationManager_INL
