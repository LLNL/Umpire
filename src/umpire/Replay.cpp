//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
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

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/DynamicPool.hpp"

#include "umpire/util/io.hpp"
#include "umpire/util/string_utils.hpp"

#include "umpire/Replay.hpp"

#include "umpire/tpl/json/json.hpp"

namespace umpire {
  
int Replay::m_argument_number = 0;

Replay::Replay() : m_replayUid(getpid())
{
  static const char* env_name = "UMPIRE_REPLAY";
  static const char* config_env_var = "UMPIRE_REPLAY_CFG";

  char* enval = getenv(env_name);
  bool enable_replay = ( enval != NULL );

  const char* config_env = std::getenv(config_env_var);
  if (config_env) {
    auto json = nlohmann::json::parse(std::string{config_env});

    std::string enabled = json["enabled"];
    if (util::case_insensitive_match(enabled, "true")) {
      enable_replay = true;
    }

    // TODO: remove once UMPIRE_REPLAY is deprecated
    if (util::case_insensitive_match(enabled, "false")) {
      std::cout << "disabling replay" << std::endl;
      enable_replay = false;
    }
  }

  replayEnabled =  enable_replay;
}

void Replay::logMessage( const std::string& message )
{
  if ( !replayEnabled )
    return;   /* short-circuit */

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

std::ostream& operator<< (std::ostream& out, umpire::Allocator& alloc) {
  out << alloc.getName();
  return out;
}

std::ostream& operator<< (
    std::ostream& out,
    umpire::strategy::DynamicPool::CoalesceHeuristic& ) {
  return out;
}

} /* namespace umpire */
