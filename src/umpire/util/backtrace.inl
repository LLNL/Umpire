//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Backtrace_INL
#define UMPIRE_Backtrace_INL

//
// Only uncomment the following line if you want to attempt to use dlopen to directly
// use the glibc version of ::backtrace instead of the overriden function (more expensive)
// that libunwind uses
// #define UMPIRE_RUN_WITH_LIBC_BACKTRACE 1

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "umpire/config.hpp"
#include "umpire/util/backtrace.hpp"

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include <cxxabi.h> // for __cxa_demangle
#endif              // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

#if !defined(_MSC_VER)
#if defined(UMPIRE_ENABLE_BACKTRACE_SYMBOLS) || defined(UMPIRE_RUN_WITH_LIBC_BACKTRACE)
#include <dlfcn.h>    // for dladdr
#include <gnu/lib-names.h>
#endif                // defined(UMPIRE_ENABLE_BACKTRACE_SYMBOLS)
#include <execinfo.h> // for backtrace
#endif                // !defined(_MSC_VER)

namespace umpire {
namespace util {

namespace {

#if defined(UMPIRE_RUN_WITH_LIBC_BACKTRACE)
using backtrace_signature = int(*)(void**, int);

inline backtrace_signature backtrace_make_func()
{
  backtrace_signature func_backtrace = nullptr;

  if (func_backtrace == nullptr) {
    // explicitly get glibc backtrace if compiled with unwind
    void* libc_handle = dlopen(LIBC_SO, RTLD_LAZY);
    if (libc_handle != nullptr) {
      dlerror(); // clear any existing errors
      // get backtrace symbol from libc library
      func_backtrace = (backtrace_signature)dlsym(libc_handle, "backtrace");
    }
    if (func_backtrace == nullptr) {
      // consume error silently
      dlerror();
      if (libc_handle != nullptr) {
        // close handle to libc
        dlclose(libc_handle);
      }
    }
    // note libc_handle is not closed if we got backtrace
  }

  if (func_backtrace == nullptr) {
    std::cout << "backtrace_make_func(): Using Global backtrace" << std::endl;
    // global backtrace
    func_backtrace = &::backtrace;
  }

  if (func_backtrace == nullptr) {
    std::cout << "backtrace_make_func(): Using Dummy backtrace" << std::endl;
    // dummy backtrace that does nothing
    func_backtrace = [](void**, int) { return 0; };
  }

  return func_backtrace;
}

inline backtrace_signature backtrace_get_func()
{
  static backtrace_signature s_backtrace = backtrace_make_func();
  return s_backtrace;
}
#endif  // defined(UMPIRE_RUN_WITH_LIBC_BACKTRACE)

bool backtrace_enabled()
{
  static bool enabled{false};
#if !defined(_WIN32)
  static bool initialized{false};

  if (!initialized) {
    const char* enval{getenv("UMPIRE_BACKTRACE")};
    if (enval) {
      std::string env_str{enval};
      std::transform(env_str.begin(), env_str.end(), env_str.begin(), ::toupper);
      if (env_str.find("ON") != std::string::npos) {
        enabled = true;
      }
    }
    initialized = true;
  }
#endif

  return enabled;
}

std::vector<void*> build_backtrace()
{
  std::vector<void*> frames;
#if !defined(_MSC_VER)
  void* callstack[128];
  const int nMaxFrames = sizeof(callstack) / sizeof(callstack[0]);
#if defined(UMPIRE_RUN_WITH_LIBC_BACKTRACE)
  const int nFrames = backtrace_get_func()(callstack, nMaxFrames);
#else
  const int nFrames = ::backtrace(callstack, nMaxFrames);
#endif  // defined(UMPIRE_RUN_WITH_LIBC_BACKTRACE)

  for (int i = 0; i < nFrames; ++i)
    frames.push_back(callstack[i]);
#endif // !defined(_MSC_VER)
  return frames;
}

std::string stringify(const std::vector<void*>& frames)
{
  std::ostringstream backtrace_stream;
#if !defined(_MSC_VER)
  int num_frames = frames.size();

  backtrace_stream << "    Backtrace: " << num_frames << " frames" << std::endl;

#if defined(UMPIRE_ENABLE_BACKTRACE_SYMBOLS)
  char** symbols = ::backtrace_symbols(&frames[0], num_frames);
  int index{0};
  for (const auto& it : frames) {
    backtrace_stream << "    " << index << " " << it << " ";
    Dl_info info;
    if (dladdr(it, &info) && info.dli_sname) {
      char* demangled = NULL;
      int status = -1;

#if !defined(_LIBCPP_VERSION)
      if (info.dli_sname[0] == '_')
        demangled = abi::__cxa_demangle(info.dli_sname, NULL, 0, &status);
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

      backtrace_stream << (status == 0 ? demangled : (info.dli_sname == 0 ? symbols[index] : info.dli_sname)) << "+0x"
                       << std::hex << static_cast<int>(static_cast<char*>(it) - static_cast<char*>(info.dli_saddr));

#if !defined(_LIBCPP_VERSION)
      free(demangled);
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
    } else {
      backtrace_stream << "No dladdr: " << symbols[index];
    }
    backtrace_stream << std::endl;
    ++index;
  }
  free(symbols);
#else // #if defined(UMPIRE_ENABLE_BACKTRACE_SYMBOLS)
  int index{0};
  for (const auto& it : frames) {
    backtrace_stream << "    " << it << std::endl;
    index++;
  }
#endif // #if defined(UMPIRE_ENABLE_BACKTRACE_SYMBOLS)

#else   // #if !defined(_MSC_VER)
  static_cast<void>(frames);
  backtrace_stream << " Backtrace not supported on Windows" << std::endl;

#endif // #if !defined(_MSC_VER)

  return backtrace_stream.str();
}

} // namespace

template <>
struct backtracer<trace_optional> {
  static void get_backtrace(backtrace& bt)
  {
    if (backtrace_enabled())
      bt.frames = build_backtrace();
  }

  static std::string print(const backtrace& bt)
  {
    if (backtrace_enabled()) {
      return stringify(bt.frames);
    } else {
      return "[UMPIRE_BACKTRACE=Off]";
    }
  }
};

template <>
struct backtracer<trace_always> {
  static void get_backtrace(backtrace& bt)
  {
    bt.frames = build_backtrace();
  }

  static std::string print(const backtrace& bt)
  {
    return stringify(bt.frames);
  }
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_Backtrace_INL
