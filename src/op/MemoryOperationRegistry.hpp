//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_OperationRegistry_HPP
#define UMPIRE_OperationRegistry_HPP

#include "umpire/op/MemoryOperation.hpp"

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Platform.hpp"

#include <memory>
#include <unordered_map>
#include <functional>

namespace umpire {
namespace op {

struct pair_hash {
  std::size_t operator () (const std::pair<Platform, Platform> &p) const {
      auto h1 = std::hash<int>{}(static_cast<int>(p.first));
      auto h2 = std::hash<int>{}(static_cast<int>(p.second));

      // Mainly for demonstration purposes, i.e. works but is overly simple
      // In the real world, use sth. like boost.hash_combine
      return h1 ^ h2;
  }
};

class MemoryOperationRegistry {
  public:

    static MemoryOperationRegistry& getInstance();

    std::shared_ptr<umpire::op::MemoryOperation> find(
        const std::string& name,
        std::shared_ptr<strategy::AllocationStrategy>& source_allocator,
        std::shared_ptr<strategy::AllocationStrategy>& dst_allocator);

    void registerOperation(
      const std::string& name,
      std::pair<Platform, Platform> platforms,
      std::shared_ptr<MemoryOperation>&& operation);

  protected:
    MemoryOperationRegistry();

  private:
    static MemoryOperationRegistry* s_memory_operation_registry_instance;

    std::unordered_map<
      std::string,
      std::unordered_map< std::pair<Platform, Platform>, 
                          std::shared_ptr<MemoryOperation>, 
                          pair_hash > > m_operators;

};

} // end of namespace op
} // end of namespace umpire

#endif
