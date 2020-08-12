//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Replay_HPP
#define UMPIRE_Replay_HPP

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/DynamicPoolMap.hpp"
#include "umpire/strategy/QuickPool.hpp"

namespace umpire {

namespace {
int m_argument_number;
}

class Allocator;

std::ostream& operator<<(std::ostream& out, umpire::Allocator&);

std::ostream& operator<<(std::ostream& out,
                         umpire::strategy::DynamicPoolMap::CoalesceHeuristic&);
std::ostream& operator<<(std::ostream& out,
                         umpire::strategy::DynamicPoolList::CoalesceHeuristic&);
std::ostream& operator<<(std::ostream& out,
                         umpire::strategy::QuickPool::CoalesceHeuristic&);

class Replay {
 public:
  void logMessage(const std::string& message);
  static Replay* getReplayLogger();
  bool replayLoggingEnabled();
  uint64_t replayUid()
  {
    return m_replayUid;
  }

  static std::string printReplayAllocator(void)
  {
    m_argument_number = 0;
    return std::string("");
  }

  template <typename T, typename... Args>
  static std::string printReplayAllocator(T&& firstArg, Args&&... args)
  {
    std::stringstream ss;

    m_argument_number++;
    if (m_argument_number != 1)
      ss << ", ";

    ss << "\"" << firstArg << "\"";

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
};

} /* namespace umpire */

#define UMPIRE_REPLAY(msg)                                                     \
  {                                                                            \
    if (umpire::Replay::getReplayLogger()->replayLoggingEnabled()) {           \
      std::ostringstream local_msg;                                            \
      auto time = std::chrono::time_point_cast<std::chrono::nanoseconds>(      \
                      std::chrono::system_clock::now())                        \
                      .time_since_epoch();                                     \
      local_msg << "{ \"kind\":\"replay\", \"uid\":"                           \
                << umpire::Replay::getReplayLogger()->replayUid() << ", "      \
                << "\"timestamp\":" << static_cast<long>(time.count()) << ", " \
                << msg << " }" << std::endl;                                   \
      umpire::Replay::getReplayLogger()->logMessage(local_msg.str());          \
    }                                                                          \
  }
#endif /* UMPIRE_Replay_HPP */
