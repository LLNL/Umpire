//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/detail/backtrace.hpp"

#include "umpire/config.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <sstream>

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include <cxxabi.h>   // for __cxa_demangle
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

#if !defined(_MSC_VER)
#include <dlfcn.h>    // for dladdr
#include <execinfo.h> // for backtrace
#endif // !defined(_MSC_VER)

namespace umpire {
namespace detail {

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
  }
#endif

  return enabled;
}

std::vector<void*> build_backtrace() {
  std::vector<void*> frames;
#if !defined(_MSC_VER)
    void *callstack[128];
    const int nMaxFrames = sizeof(callstack) / sizeof(callstack[0]);
    for (int i = 0; i < ::backtrace(callstack, nMaxFrames); ++i)
      frames.push_back(callstack[i]);
#endif // !defined(_MSC_VER)
    return frames;
}

std::string stringify_backtrace(const std::vector<void*>& frames)
{
  std::ostringstream backtrace_stream;
#if !defined(_MSC_VER)
  int num_frames = frames.size();
  char** symbols = ::backtrace_symbols(&frames[0], num_frames);

  backtrace_stream << "    Backtrace: " << num_frames << " frames" << std::endl;

  int index{0};
  for (const auto& it : frames) {
    backtrace_stream << "    " << index << " " << it << " "; 
    Dl_info info;
    if (dladdr(it, &info) && info.dli_sname) {
      char *demangled = NULL;
      int status = -1;
#if !defined(_LIBCPP_VERSION)
      if (info.dli_sname[0] == '_')
        demangled = abi::__cxa_demangle(info.dli_sname, NULL, 0, &status);
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

      backtrace_stream << ( status == 0 ? demangled : ( info.dli_sname == 0 ? symbols[index] : info.dli_sname ) )
        << "+0x" << std::hex << static_cast<int>(static_cast<char*>(it) - static_cast<char*>(info.dli_saddr));

#if !defined(_LIBCPP_VERSION)
      free(demangled);
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
    }
    else {
      backtrace_stream << "No dladdr: " << symbols[index];
    }
    backtrace_stream << std::endl;
    ++index;
  }
  free(symbols);
#else
  static_cast<void>(frames);
  backtrace_stream << " Backtrace not supported on Windows" << std::endl;
#endif
  return backtrace_stream.str();
}

} // end of namespace detail
} // end of namespace umpire