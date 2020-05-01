//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_NamedAllocator_HPP
#define UMPIRE_NamedAllocator_HPP

#include "umpire/Allocator.hpp"

#include <string>
#include <unordered_map>

namespace umpire {

class NamedAllocator {
  public:
    NamedAllocator(Allocator alloc);
    void* allocate(std::string name, std::size_t bytes);
    void* get_allocation_by_name(std::string name);
    void deallocate(void* ptr);

  private:
    Allocator m_allocator;
    std::unordered_map<std::string, void*> m_names_to_allocations;
    std::unordered_map<void*, std::string> m_allocations_to_names;
};

} // end of namespace umpire

#endif // UMPIRE_NamedAllocator_HPP