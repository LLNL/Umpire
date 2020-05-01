//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/NamedAllocator.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {

NamedAllocator::NamedAllocator(Allocator allocator):
    m_allocator(allocator)
{
}

void*
NamedAllocator::allocate(std::string name, std::size_t bytes)
{
    void* ptr = m_allocator.allocate(bytes);
    m_names_to_allocations[name] = ptr;
    m_allocations_to_names[ptr] = name;
    return ptr;
}

void
NamedAllocator::deallocate(void* ptr)
{
    m_allocator.deallocate(ptr);

    m_names_to_allocations.erase(m_allocations_to_names[ptr]);
    m_allocations_to_names.erase(ptr);
}

void*
NamedAllocator::get_allocation_by_name(std::string name)
{
    return m_names_to_allocations[name];
}

} // end of namespace umpire
