//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SharedAllocationStrategy_HPP
#define UMPIRE_SharedAllocationStrategy_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

#include <string>

namespace umpire {
namespace strategy {

/*!
 * \brief SharedAllocationStrategy provides a unified interface to all classes that
 * can be used to allocate and free data.
 */
class SharedAllocationStrategy : public AllocationStrategy
{
  public:
    SharedAllocationStrategy(const std::string& name, int id) noexcept;

    virtual ~SharedAllocationStrategy() = default;

    virtual void* allocate(std::size_t bytes);
    virtual void* allocate(std::string name, std::size_t bytes);

    virtual void* get_allocation_by_name(std::string allocation_name);

    virtual void set_foreman(int id) = 0;
    virtual bool is_foreman() = 0;
    virtual void synchronize() = 0;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_SharedAllocationStrategy_HPP
