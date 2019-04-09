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

#include <string>
#include <sstream>
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/DynamicPool.hpp"

namespace umpire {
namespace replay {
std::ostream& operator<< (std::ostream& out, umpire::Allocator& );
std::ostream& operator<< (std::ostream& out, 
    umpire::strategy::DynamicPool::Coalesce_Heuristic& );
class Replay {
public:
  static void initialize();
  static void finalize();

  void logMessage( const std::string& message );
  static Replay* getReplayLogger();
  bool replayLoggingEnabled();
  uint64_t replayUid() { return m_replayUid; }

  static std::string printReplayAllocator( void ) {
    return std::string("");
  }

  template <typename T, typename... Args>
  static std::string printReplayAllocator(T&& firstArg, Args&&... args) {
    std::stringstream ss;

    if (typeid(firstArg) != typeid(umpire::strategy::DynamicPool::Coalesce_Heuristic))
      ss << "," << firstArg;

    ss << printReplayAllocator(std::forward<Args>(args)...);
    return ss.str();
  }
private:
  Replay(bool enable_replay);
  ~Replay();

  bool replayEnabled;
  uint64_t m_replayUid;
  static Replay* s_Replay;
};

} /* namespace replay */
} /* namespace umpire */

#define UMPIRE_REPLAY( msg )                                                 \
{                                                                            \
  if (umpire::replay::Replay::getReplayLogger()->replayLoggingEnabled()) {   \
    std::ostringstream local_msg;                                            \
    local_msg                                                                \
      << "REPLAY,"                                                           \
      << umpire::replay::Replay::getReplayLogger()->replayUid() << ","       \
      << msg                                                                 \
      << std::endl;                                                          \
    umpire::replay::Replay::getReplayLogger()->logMessage(local_msg.str());  \
  }                                                                          \
}
#endif /* UMPIRE_Replay_HPP */
