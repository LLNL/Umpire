//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_wrap_allocator_HPP
#define UMPIRE_wrap_allocator_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace util {

template <typename Base>
std::unique_ptr<Base>
do_wrap(std::unique_ptr<Base>&& allocator)
{
  return std::move(allocator);
}


template <typename Base, typename Strategy, typename... Strategies>
std::unique_ptr<Base>
do_wrap(std::unique_ptr<Base>&& allocator)
{
  return std::unique_ptr<Base>(new Strategy(umpire::util::do_wrap<Base, Strategies...>(std::move(allocator))));
  //return util::do_wrap<Base, Strategies...>(std::unique_ptr<Base>{new Strategy(std::move(strategy))});
}

template<typename... Strategies>
std::unique_ptr<strategy::AllocationStrategy>
wrap_allocator(std::unique_ptr<strategy::AllocationStrategy>&& allocator)
{
  return umpire::util::do_wrap<
    umpire::strategy::AllocationStrategy, Strategies...>(
      std::move(allocator));
}

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_wrap_allocator_HPP
