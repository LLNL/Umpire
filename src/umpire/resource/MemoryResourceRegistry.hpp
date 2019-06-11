//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#ifndef UMPIRE_MemoryResourceRegistry_HPP
#define UMPIRE_MemoryResourceRegistry_HPP

#include "umpire/resource/MemoryResource.hpp"
#include "umpire/resource/MemoryResourceFactory.hpp"

#include <memory>
#include <vector>

namespace umpire {
namespace resource {

class MemoryResourceRegistry {
  public:
    static MemoryResourceRegistry& getInstance() noexcept;

    resource::MemoryResource* makeMemoryResource(const std::string& name, int id);

    void registerMemoryResource(std::unique_ptr<MemoryResourceFactory>&& factory);

    MemoryResourceRegistry(const MemoryResourceRegistry&) = delete;
    MemoryResourceRegistry& operator=(const MemoryResourceRegistry&) = delete;
    ~MemoryResourceRegistry() = default;

  private:
    MemoryResourceRegistry() noexcept;

    std::vector<std::unique_ptr<MemoryResourceFactory> > m_allocator_factories;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_MemoryResourceRegistry_HPP
