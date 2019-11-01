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
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
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
template<typename Strategy, bool Introspection, typename... Args>
void ReplayOperationManager::makeAdvisor(
    const std::string allocator_name,
    const std::string base_allocator_name,
    Args&&... args
)
{
  m_cont_op = new ReplayOperation;

  m_cont_op->op = [=]() {
    auto& rm = umpire::ResourceManager::getInstance();

    this->m_allocator_array.push_back(
      rm.makeAllocator<Strategy, Introspection> (
        allocator_name, rm.getAllocator(base_allocator_name), std::forward<Args>(args) ...)
    );
  };

  operations.push_back(m_cont_op);
}

template<typename Strategy, bool Introspection, typename... Args>
void ReplayOperationManager::makeAdvisor(
    const std::string allocator_name,
    const std::string base_allocator_name,
    const std::string advice_operation,
    const std::string accessing_allocator_name,
    Args&&... args
)
{
  m_cont_op = new ReplayOperation;

  m_cont_op->op = [=]() {
    auto& rm = umpire::ResourceManager::getInstance();

    this->m_allocator_array.push_back(
      rm.makeAllocator<Strategy, Introspection>
        (   allocator_name
          , rm.getAllocator(base_allocator_name)
          , advice_operation
          , rm.getAllocator(accessing_allocator_name)
          , std::forward<Args>(args)...
        )
    );
  };

  operations.push_back(m_cont_op);
}
#endif // REPLAY_ReplayOperationManager_INL
