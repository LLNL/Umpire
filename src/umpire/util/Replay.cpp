//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#include "umpire/util/Replay.hpp"

#include <iostream>   // for std::cout, std::cerr
#include <stdlib.h>   // for getenv()
#include <strings.h>  // for strcasecmp()

namespace umpire {
namespace util {

static const char* env_name = "UMPIRE_REPLAY";
Replay* Replay::s_Replay = nullptr;

Replay::Replay(bool enable_replay) : replayEnabled(enable_replay)
{
}

Replay::~Replay()
{
}

void Replay::logMessage( const std::string& message )
{
  if ( !replayEnabled )
    return;   /* short-circuit */

  std::cout << "REPLAY " << message << std::endl;
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
  bool enable_replay = ( enval != NULL );

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

} /* namespace util */
} /* namespace umpire */
