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

//
// AllocationAdvisor
//
template <typename... Args>
void ReplayOperation::makeAdvisor(
    const bool introspection,
    const std::string& allocator_name,
    const std::string& base_allocator_name,
    Args&&... args
)
{
  if (introspection) {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::AllocationAdvisor, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args)...
          )
      );
    };
  }
  else {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::AllocationAdvisor, false>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args)...
          )
      );
    };
  }
}

template <typename... Args>
void ReplayOperation::makeAdvisor(
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

      this->m_my_manager.m_allocator_array.push_back(
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
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
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
}

//
// FixedPool
//
template<typename... Args>
void ReplayOperation::makeFixedPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , Args&&... args
)
{
  if (introspection) {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::FixedPool, true>
          (  allocator_name
           , rm.getAllocator(base_allocator_name)
           , std::forward<Args>(args)...
          )
      );
    };
  }
  else {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::FixedPool, false>
          (  allocator_name
           , rm.getAllocator(base_allocator_name)
           , std::forward<Args>(args)...
          )
      );
    };
  }
}

//
// DynamicPool
//
template <typename... Args>
void ReplayOperation::makeDynamicPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , Args&&... args
)
{
  if (introspection) {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::DynamicPool, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args)...
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
            , std::forward<Args>(args)...
          )
      );
    };
  }
}

template <typename... Args>
void ReplayOperation::makeMixedPool(
    const bool introspection
  , const std::string& allocator_name
  , const std::string& base_allocator_name
  , Args&&... args
)
{
  if (introspection) {
    op = [=]() {
      auto& rm = umpire::ResourceManager::getInstance();

      this->m_my_manager.m_allocator_array.push_back(
        rm.makeAllocator<umpire::strategy::MixedPool, true>
          (   allocator_name
            , rm.getAllocator(base_allocator_name)
            , std::forward<Args>(args)...
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
            , std::forward<Args>(args)...
          )
      );
    };
  }
}
#endif // REPLAY_ReplayOperationManager_INL
