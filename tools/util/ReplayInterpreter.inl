//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_ReplayInterpreter_INL
#define REPLAY_ReplayInterpreter_INL

#include <sstream>

template <typename T> void
ReplayInterpreter::get_from_string( const std::string& s, T& val )
{
    std::istringstream ss(s);
    ss >> val;
}

#endif // REPLAY_ReplayInterpreter_INL
