//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_Macros_HPP
#define REPLAY_Macros_HPP

#include <cstdlib>
#include <iostream>

#define REPLAY_ERROR( msg )                                                 \
{                                                                           \
  std::cerr << std::string(__FILE__) << " " << __LINE__ << " " << __func__  \
    << " " << msg << std::endl;                                             \
  exit(-1);                                                                 \
}

#endif // REPLAY_Macros_HPP
