//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_ReplayInterpreter_INL
#define REPLAY_ReplayInterpreter_INL

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include <sstream>

template <typename T> void
ReplayInterpreter::get_from_string( const std::string& s, T& val )
{
    std::istringstream ss(s);
    ss >> val;
}
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

#endif // REPLAY_ReplayInterpreter_INL
