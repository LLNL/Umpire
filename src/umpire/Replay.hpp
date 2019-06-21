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
#ifndef UMPIRE_Replay_HPP
#define UMPIRE_Replay_HPP

#include <chrono>
#include <string>
#include <sstream>
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/DynamicPool.hpp"

namespace umpire {
std::ostream& operator<< (std::ostream& out, umpire::Allocator& );
std::ostream& operator<< (std::ostream& out,
    umpire::strategy::DynamicPool::CoalesceHeuristic& );
class Replay {
public:
  void logMessage( const std::string& message );
  static Replay* getReplayLogger();
  bool replayLoggingEnabled();
  uint64_t replayUid() { return m_replayUid; }

  static std::string printReplayAllocator( void ) {
    m_argument_number = 0;
    return std::string("");
  }

  template <typename T, typename... Args>
  static std::string printReplayAllocator(T&& firstArg, Args&&... args) {
    std::stringstream ss;

    if (typeid(firstArg) != typeid(umpire::strategy::DynamicPool::CoalesceHeuristic)) {
      m_argument_number++;
      if ( m_argument_number != 1 )
        ss << ", ";

      ss << "\"" << firstArg << "\"";
    }

    ss << printReplayAllocator(std::forward<Args>(args)...);
    return ss.str();
  }
private:
  Replay();
  ~Replay() = default;

  Replay(const Replay&) = delete;
  Replay& operator=(const Replay&) = delete;

  bool replayEnabled;
  uint64_t m_replayUid;
  static int m_argument_number;
};

} /* namespace umpire */

#define UMPIRE_REPLAY( msg )                                                 \
{                                                                            \
  if (umpire::Replay::getReplayLogger()->replayLoggingEnabled()) {   \
    std::ostringstream local_msg;                                            \
    auto time = std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now()).time_since_epoch();\
    local_msg                                                                \
      << "{ \"kind\":\"replay\", \"uid\":"                                   \
      << umpire::Replay::getReplayLogger()->replayUid() << ", "      \
      << "\"timestamp\":"                                                    \
      << static_cast<long>(time.count()) << ", "                             \
      << msg                                                                 \
      << " }"                                                                \
      << std::endl;                                                          \
    umpire::Replay::getReplayLogger()->logMessage(local_msg.str());  \
  }                                                                          \
}
#endif /* UMPIRE_Replay_HPP */
