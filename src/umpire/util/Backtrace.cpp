//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <cstdio>
#include <cstdlib>

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include <cxxabi.h>   // for __cxa_demangle
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

#if !defined(_MSC_VER)
#include <dlfcn.h>    // for dladdr
#include <execinfo.h> // for backtrace
#endif // !defined(_MSC_VER)

#include <iostream>
#include <iomanip>

#include "umpire/util/Backtrace.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace util {

Backtrace::Backtrace() noexcept
{
}

void Backtrace::getBacktrace()
{
#if !defined(_MSC_VER)
  void *callstack[128];
  const int nMaxFrames = sizeof(callstack) / sizeof(callstack[0]);

  for (int i = 0; i < backtrace(callstack, nMaxFrames); ++i)
    m_backtrace.push_back(callstack[i]);
#endif // !defined(_MSC_VER)
}

std::ostream& operator<<(std::ostream& os, const Backtrace& bt)
{
#if !defined(_MSC_VER)
  char **symbols = backtrace_symbols(&bt.m_backtrace[0], bt.m_backtrace.size());

  os << "    Backtrace: " << bt.m_backtrace.size() << " frames" << std::endl;
  int index = 0;
  for ( const auto& it : bt.m_backtrace ) {
    os << "    " << index << " " << it << " "; 

    Dl_info info;
    if (dladdr(it, &info) && info.dli_sname) {
      char *demangled = NULL;
      int status = -1;
#if !defined(_LIBCPP_VERSION)
      if (info.dli_sname[0] == '_')
        demangled = abi::__cxa_demangle(info.dli_sname, NULL, 0, &status);
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

      os << ( status == 0 ? demangled : ( info.dli_sname == 0 ? symbols[index] : info.dli_sname ) )
        << "+0x" << std::hex << static_cast<int>(static_cast<char*>(it) - static_cast<char*>(info.dli_saddr));

#if !defined(_LIBCPP_VERSION)
      free(demangled);
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
    }
    else {
      os << "No dladdr: " << symbols[index];
    }
    os << std::endl;
    ++index;
  }
  free(symbols);
#else
  UMPIRE_USE_VAR(bt);
  os << " Backtrace not supported on Windows platform" << std::endl;
#endif
  return os;
}

} // end of namespace util
} // end of namespace umpire
