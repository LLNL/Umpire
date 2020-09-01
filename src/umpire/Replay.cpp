//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <stdlib.h> // for getenv()

#include <iostream> // for std::cout, std::cerr

#if !defined(_MSC_VER)
#include <strings.h> // for strcasecmp()
#include <unistd.h>  // getpid()
#else
#include <process.h>
#define strcasecmp _stricmp
#define getpid _getpid
#endif

#include "umpire/Allocator.hpp"
#include "umpire/Replay.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/util/io.hpp"

namespace umpire {

static const char* env_name = "UMPIRE_REPLAY";

Replay::Replay() : m_replayUid(getpid())
{
  char* enval = getenv(env_name);
  bool enable_replay = (enval != NULL);

  replayEnabled = enable_replay;
}

void Replay::logMessage(const std::string& message)
{
  if (!replayEnabled)
    return; /* short-circuit */

  umpire::replay() << message;
}

bool Replay::replayLoggingEnabled()
{
  return replayEnabled;
}

Replay* Replay::getReplayLogger()
{
  static Replay replay_logger;

  return &replay_logger;
}

std::ostream& operator<<(std::ostream& out, umpire::Allocator& alloc)
{
  out << alloc.getName();
  return out;
}

std::ostream& operator<<(std::ostream& out,
                         umpire::strategy::DynamicPoolMap::CoalesceHeuristic&)
{
  return out;
}

std::ostream& operator<<(std::ostream& out,
                         umpire::strategy::DynamicPoolList::CoalesceHeuristic&)
{
  return out;
}

std::ostream& operator<<(std::ostream& out,
                         umpire::strategy::QuickPool::CoalesceHeuristic&)
{
  return out;
}

} /* namespace umpire */
