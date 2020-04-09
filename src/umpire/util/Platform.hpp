//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Platform_HPP
#define UMPIRE_Platform_HPP

#if defined(_WIN32) && !defined(UMPIRESTATICLIB)
#ifdef UMPIRESHAREDDLL_EXPORTS
#define UMPIRESHAREDDLL_API __declspec(dllexport)
#else
#define UMPIRESHAREDDLL_API __declspec(dllimport)
#endif
#else
#define UMPIRESHAREDDLL_API
#endif

#include "camp/resource/platform.hpp"

namespace umpire {

using Platform = camp::resources::Platform;

} // end of namespace umpire

#endif
