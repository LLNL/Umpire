//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include <vector>
#include <string>

namespace umpire {
namespace detail {

bool backtrace_enabled();
std::vector<void*> build_backtrace();
std::string stringify_backtrace(const std::vector<void*>& frames);

struct backtrace{
  std::vector<void*> frames;
};

struct trace_optional {};
struct trace_always {};

template<typename TraceType=trace_optional>
struct backtracer {};

template<>
struct backtracer<trace_optional>
{
  static void get_backtrace(backtrace& bt) {
    if (backtrace_enabled())
      bt.frames = build_backtrace();
  }

  static std::string print(const backtrace& bt) {
    if (backtrace_enabled()) {
      return stringify_backtrace(bt.frames);
    } else {
      return "[UMPIRE_BACKTRACE=Off]";
    }
  }
};

template<>
struct backtracer<trace_always>
{
  static void get_backtrace(backtrace& bt) {
    bt.frames = build_backtrace();
  }

  static std::string print(const backtrace& bt) {
    return stringify_backtrace(bt.frames);
  }
};

} // end of namespace detail
} // end of namespace umpire