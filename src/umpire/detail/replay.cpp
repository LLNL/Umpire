//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <iostream>   // for std::cout, std::cerr
#include <stdlib.h>   // for getenv()

#if !defined(_MSC_VER)
#include <strings.h>  // for strcasecmp()
#include <unistd.h>   // getpid()
#else
#include <process.h>
#define strcasecmp _stricmp
#define getpid _getpid
#endif

#include "umpire/allocator.hpp"
#include "umpire/memory.hpp"
#include "umpire/detail/io.hpp"

#include "umpire/detail/replay.hpp"

namespace umpire {
namespace detail {

static const char* env_name = "UMPIRE_REPLAY";
int replay::m_argument_number = 0;

replay::replay() : m_replayUid(getpid())
{
  char* enval = getenv(env_name);
  bool enable_replay = ( enval != NULL );

  replayEnabled =  enable_replay;
}

void replay::logMessage( const std::string& message )
{
  if ( !replayEnabled )
    return;   /* short-circuit */

  umpire::replay() << message;
}

bool replay::replayLoggingEnabled()
{
  return replayEnabled;
}

replay* replay::getReplayLogger()
{
  static replay replay_logger;

  return &replay_logger;
}

}
}
