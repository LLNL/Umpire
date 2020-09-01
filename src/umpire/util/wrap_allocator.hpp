//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_wrap_allocator_HPP
#define UMPIRE_wrap_allocator_HPP

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/AllocationTracker.hpp"
#include "umpire/strategy/ZeroByteHandler.hpp"
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
  return std::unique_ptr<Base>(new Strategy(
      umpire::util::do_wrap<Base, Strategies...>(std::move(allocator))));
}

template <typename... Strategies>
std::unique_ptr<strategy::AllocationStrategy> wrap_allocator(
    std::unique_ptr<strategy::AllocationStrategy>&& allocator)
{
  return umpire::util::do_wrap<umpire::strategy::AllocationStrategy,
                               Strategies...>(std::move(allocator));
}

template <typename Strategy>
Strategy* unwrap_allocation_strategy(
    strategy::AllocationStrategy* base_strategy)
{
  umpire::strategy::ZeroByteHandler* zero{nullptr};
  umpire::strategy::AllocationTracker* tracker{nullptr};
  Strategy* strategy{nullptr};

  tracker = dynamic_cast<umpire::strategy::AllocationTracker*>(base_strategy);

  if (tracker) {
    zero = dynamic_cast<umpire::strategy::ZeroByteHandler*>(
        tracker->getAllocationStrategy());
  } else {
    zero = dynamic_cast<umpire::strategy::ZeroByteHandler*>(base_strategy);
  }

  if (zero) {
    strategy = dynamic_cast<Strategy*>(zero->getAllocationStrategy());
  } else {
    if (tracker) {
      strategy = dynamic_cast<Strategy*>(tracker->getAllocationStrategy());
    } else {
      strategy = dynamic_cast<Strategy*>(base_strategy);
    }
  }

  if (!strategy) {
    UMPIRE_ERROR("Couldn't unwrap " << base_strategy->getName() << " to "
                                    << typeid(Strategy).name());
  }

  return strategy;
}

template <typename Strategy>
Strategy* unwrap_allocator(Allocator allocator)
{
  return unwrap_allocation_strategy<Strategy>(
      allocator.getAllocationStrategy());
}

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_wrap_allocator_HPP
