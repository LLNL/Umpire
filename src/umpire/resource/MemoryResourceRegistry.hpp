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
#include <list>

namespace umpire {
namespace resource {

class MemoryResourceRegistry {
  public:
    static MemoryResourceRegistry& getInstance() noexcept;

    std::shared_ptr<umpire::resource::MemoryResource> makeMemoryResource(const std::string& name, int id,
                                                                         const MemoryResourceTraits traits = MemoryResourceTraits{});

    void registerMemoryResource(std::shared_ptr<MemoryResourceFactory>&& factory);

  protected:
    MemoryResourceRegistry() noexcept;

  private:
    static MemoryResourceRegistry* s_allocator_registry_instance;

    std::list<std::shared_ptr<MemoryResourceFactory> > m_allocator_factories;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_MemoryResourceRegistry_HPP
