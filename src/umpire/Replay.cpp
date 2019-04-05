//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////

#include <iostream>   // for std::cout, std::cerr
#include <cstdlib>   // for getenv()
#include <strings.h>  // for strcasecmp()

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/DynamicPool.hpp"

#include "umpire/Replay.hpp"

namespace umpire {
namespace replay {

static const char* env_name = "UMPIRE_REPLAY";
Replay* Replay::s_Replay = nullptr;

Replay::Replay(bool enable_replay) : replayEnabled(enable_replay)
{
}

Replay::~Replay()
= default;

void Replay::logMessage( const std::string& message )
{
  if ( !replayEnabled )
    return;   /* short-circuit */

  std::cout << message;
}

bool Replay::replayLoggingEnabled()
{
  return replayEnabled;
}

void Replay::initialize()
{
  if ( s_Replay != nullptr )
    return;

  char* enval = getenv(env_name);
  bool enable_replay = ( enval != nullptr );

  s_Replay = new Replay(enable_replay);
}

void Replay::finalize()
{
  delete s_Replay;
  s_Replay = nullptr;
}

Replay* Replay::getReplayLogger()
{
  if ( s_Replay == nullptr )
    Replay::initialize();

  return s_Replay;
}

std::ostream& operator<< (std::ostream& out, umpire::Allocator& alloc) {
  out << alloc.getName();
  return out;
}

std::ostream& operator<< (
    std::ostream& out, 
    umpire::strategy::DynamicPool::Coalesce_Heuristic& ) {
  return out;
}

} /* namespace replay */
} /* namespace umpire */
