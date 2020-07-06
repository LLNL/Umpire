//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_Macros_HPP
#define REPLAY_Macros_HPP
#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

#include <cstdlib>
#include <iostream>

#define REPLAY_WARNING( msg )                                               \
{                                                                           \
  std::cout << std::string(__FILE__) << " " << __LINE__ << " " << __func__  \
    << " " << msg << std::endl;                                             \
}       

#define REPLAY_ERROR( msg )                                                 \
{                                                                           \
  std::cerr << std::string(__FILE__) << " " << __LINE__ << " " << __func__  \
    << " " << msg << std::endl;                                             \
  exit(-1);                                                                 \
}

#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#endif // REPLAY_Macros_HPP
