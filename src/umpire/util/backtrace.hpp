//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Backtrace_HPP
#define UMPIRE_Backtrace_HPP

#include <vector>

namespace umpire {
namespace util {

struct backtrace {
  std::vector<void*> frames;
};

struct trace_optional {
};
struct trace_always {
};

template <typename TraceType = trace_optional>
struct backtracer {
};

} // end of namespace util
} // end of namespace umpire

#include "umpire/util/backtrace.inl"

#endif // UMPIRE_Backtrace_HPP
