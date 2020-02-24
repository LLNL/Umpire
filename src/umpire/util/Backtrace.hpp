//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Backtrace_HPP
#define UMPIRE_Backtrace_HPP

#include <iostream>
#include <vector>

namespace umpire {
namespace util {

class Backtrace
{
public:
  Backtrace() noexcept;
  void getBacktrace();

  friend std::ostream& operator<<(std::ostream& os, const Backtrace& bt);
private:
  std::vector<void*> m_backtrace;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_Backtrace_HPP
