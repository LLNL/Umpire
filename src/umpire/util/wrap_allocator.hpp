//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_wrap_allocator_HPP
#define UMPIRE_wrap_allocator_HPP

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace util {

template <typename Base>
std::unique_ptr<Base> do_wrap(std::unique_ptr<Base>&& allocator)
{
  return std::move(allocator);
}

template <typename Base, typename Strategy, typename... Strategies>
std::unique_ptr<Base> do_wrap(std::unique_ptr<Base>&& allocator)
{
  return std::unique_ptr<Base>(new Strategy(umpire::util::do_wrap<Base, Strategies...>(std::move(allocator))));
}

template <typename... Strategies>
std::unique_ptr<strategy::AllocationStrategy> wrap_allocator(std::unique_ptr<strategy::AllocationStrategy>&& allocator)
{
  return umpire::util::do_wrap<umpire::strategy::AllocationStrategy, Strategies...>(std::move(allocator));
}

template <typename Strategy>
Strategy* unwrap_allocation_strategy(strategy::AllocationStrategy* base_strategy)
{
  if (!base_strategy) {
    UMPIRE_ERROR(runtime_error, fmt::format("Cannot unwrap null allocator to strategy \"{}\"",
                                            typeid(Strategy).name()));
  }

  Strategy* strategy{dynamic_cast<Strategy*>(base_strategy)};

  if (!strategy) {
    UMPIRE_ERROR(runtime_error, fmt::format("Couldn't unwrap allocator \"{}\" to strategy \"{}\"",
                                            base_strategy->getName(), typeid(Strategy).name()));
  }

  return strategy;
}

template <typename Strategy>
Strategy* unwrap_allocator(Allocator allocator)
{
  return unwrap_allocation_strategy<Strategy>(allocator.getAllocationStrategy());
}

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_wrap_allocator_HPP
