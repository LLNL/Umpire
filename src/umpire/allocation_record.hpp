//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/config.hpp"

#include <cstddef>
#include <memory>

#if defined(UMPIRE_ENABLE_BACKTRACE)
#include "umpire/util/backtrace.hpp"
#endif

namespace umpire {

class memory;

struct allocation_record
{
  void* ptr{nullptr};
  std::size_t size{0};
  memory* strategy{nullptr};
#if defined(UMPIRE_ENABLE_BACKTRACE)
  util::backtrace allocation_backtrace;
#endif // UMPIRE_ENABLE_BACKTRACE
};

} // end of namespace umpire
