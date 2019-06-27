//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_wrap_allocator_HPP
#define UMPIRE_wrap_allocator_HPP

#include "umpire/util/make_unique.hpp"

#include "umpire/strategy/AllocationStrategy.hpp"


namespace umpire {
namespace util {

template <typename Base, typename Strategy, typename... Strategies>
std::unique_ptr<Base>
do_wrap(std::unique_ptr<Base>&& strategy)
{
  return std::unique_ptr<Base>(new Strategy(do_wrap<Base, Strategies...>(std::move(strategy))));
  //return util::do_wrap<Base, Strategies...>(std::unique_ptr<Base>{new Strategy(std::move(strategy))});
}

template <typename Base>
std::unique_ptr<Base>
do_wrap(std::unique_ptr<Base>&& strategy)
{
  return std::move(strategy);
}


template<typename... Strategies>
std::unique_ptr<strategy::AllocationStrategy>
wrap_allocator(std::unique_ptr<strategy::AllocationStrategy>&& allocator)
{
  return do_wrap<strategy::AllocationStrategy, Strategies...>(
      std::move(allocator));
}

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_wrap_allocator_HPP
