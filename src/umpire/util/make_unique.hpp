//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_make_unique_HPP
#define UMPIRE_make_unique_HPP

#include <memory>

namespace umpire {
namespace util {

template <typename T, typename... Args>
constexpr std::unique_ptr<T> make_unique(Args&&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // end of namespace util
} // end of namespace umpire

#endif
