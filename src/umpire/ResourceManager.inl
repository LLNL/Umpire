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
#ifndef UMPIRE_ResourceManager_INL
#define UMPIRE_ResourceManager_INL

#include "umpire/ResourceManager.hpp"

#include <sstream>

#include "umpire/util/Macros.hpp"
#include "umpire/strategy/AllocationTracker.hpp"
#ifndef _MSC_VER
#include <cxxabi.h>
#endif

namespace umpire {

template <typename T, typename... Args>
std::string ResourceManager::printReplayAllocator(
    T&& firstArg,
    Args&&... args
)
{
  std::stringstream ss;

  ss << ", ???" << abi::__cxa_demangle(typeid(firstArg).name(), nullptr, nullptr, nullptr) << "???";

  ss << printReplayAllocator(std::forward<Args>(args)...);
  return ss.str();
}

template <typename... Args>
std::string ResourceManager::printReplayAllocator(
    int&& firstArg,
    Args&&... args
)
{
  std::stringstream ss;

  ss << ", " << firstArg;

  ss << printReplayAllocator(std::forward<Args>(args)...);
  return ss.str();
}

template <typename... Args>
std::string ResourceManager::printReplayAllocator(
    umpire::Allocator&& firstArg,
    Args&&... args
)
{
  std::stringstream ss;

  ss << ", rm.getAllocator(\"" << firstArg.getName() << "\")";

  ss << printReplayAllocator(std::forward<Args>(args)...);
  return ss.str();
}

template <typename Strategy,
         bool introspection,
         typename... Args>
Allocator ResourceManager::makeAllocator(
    const std::string& name, 
    Args&&... args)
{
  std::shared_ptr<strategy::AllocationStrategy> allocator;

  try {
    UMPIRE_LOCK;

    UMPIRE_REPLAY("<" << 
        abi::__cxa_demangle(typeid(Strategy).name(), nullptr, nullptr, nullptr)
        << ", " << (introspection ? "true" : "false")  << ">("
        << "\"" << name << "\""
        << printReplayAllocator(std::forward<Args>(args)...)
        <<");\n"
    );

    UMPIRE_LOG(Debug, "(name=\"" << name << "\")");

    if (isAllocator(name)) {
      UMPIRE_ERROR("Allocator with name " << name << " is already registered.");
    }

    if (!introspection) {
      allocator = std::make_shared<Strategy>(name, getNextId(), std::forward<Args>(args)...);

      m_allocators_by_name[name] = allocator;
      m_allocators_by_id[allocator->getId()] = allocator;
    } else {
      std::stringstream base_name;
      base_name << name << "_base";


      auto base_allocator = std::make_shared<Strategy>(base_name.str(), getNextId(), std::forward<Args>(args)...);

      allocator = std::make_shared<umpire::strategy::AllocationTracker>(name, getNextId(), Allocator(base_allocator));

      m_allocators_by_name[name] = allocator;
      m_allocators_by_id[allocator->getId()] = allocator;

    }

    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }

  return Allocator(allocator);
}

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_INL
