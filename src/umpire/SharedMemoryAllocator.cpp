//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/SharedMemoryAllocator.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {

SharedMemoryAllocator::SharedMemoryAllocator(Allocator allocator):
    m_allocator(allocator)
{
}

void*
SharedMemoryAllocator::allocate(std::string name, std::size_t bytes)
{
    void* ptr = m_allocator.allocate(bytes);
    m_names_to_allocations[name] = ptr;
    m_allocations_to_names[ptr] = name;
    return ptr;
}

void
SharedMemoryAllocator::deallocate(void* ptr)
{
    m_allocator.deallocate(ptr);

    m_names_to_allocations.erase(m_allocations_to_names[ptr]);
    m_allocations_to_names.erase(ptr);
}

void*
SharedMemoryAllocator::get_allocation_by_name(std::string name)
{
    return m_names_to_allocations[name];
}

} // end of namespace umpire
