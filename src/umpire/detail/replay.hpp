//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include <chrono>
#include <string>
#include <sstream>
#include <iostream>

namespace umpire {
namespace detail {

class replay {
public:
  void logMessage( const std::string& message );
  static replay* getReplayLogger();
  bool replayLoggingEnabled();
  uint64_t replayUid() { return m_replayUid; }

  static std::string printReplayAllocator( void ) {
    m_argument_number = 0;
    return std::string("");
  }

  template <typename T, typename... Args>
  static std::string printReplayAllocator(T&& firstArg, Args&&... args) {
    std::stringstream ss;

    m_argument_number++;
    if ( m_argument_number != 1 )
      ss << ", ";

    ss << "\"" << firstArg << "\"";

    ss << printReplayAllocator(std::forward<Args>(args)...);
    return ss.str();
  }
private:
  replay();
  ~replay() = default;

  replay(const replay&) = delete;
  replay& operator=(const replay&) = delete;

  bool replayEnabled;
  uint64_t m_replayUid;
  static int m_argument_number;
};

}
}

#define UMPIRE_REPLAY( msg )                                                 \
{                                                                            \
  if (umpire::detail::replay::getReplayLogger()->replayLoggingEnabled()) {   \
    std::ostringstream local_msg;                                            \
    auto time = std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now()).time_since_epoch();\
    local_msg                                                                \
      << "{ \"kind\":\"replay\", \"uid\":"                                   \
      << umpire::detail::replay::getReplayLogger()->replayUid() << ", "      \
      << "\"timestamp\":"                                                    \
      << static_cast<long>(time.count()) << ", "                             \
      << msg                                                                 \
      << " }"                                                                \
      << std::endl;                                                          \
    umpire::detail::replay::getReplayLogger()->logMessage(local_msg.str());  \
  }                                                                          \
}
