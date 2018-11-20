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
#ifndef UMPIRE_Replay_HPP
#define UMPIRE_Replay_HPP

#include <string>
#include <sstream>
#include "umpire/Allocator.hpp"
#include <cxxabi.h>

namespace umpire {
namespace util {
class Replay {
  public:

  void logMessage( const std::string& message );

  static void initialize();

  static void finalize();

  static Replay* getReplayLogger();

  bool replayLoggingEnabled();

  static std::string printReplayAllocator( void ) {
    return std::string("");
  }

  template <typename T, typename... Args>
  static std::string printReplayAllocator(
    T&& firstArg,
    Args&&... args
  ) {
    std::stringstream ss;

    ss << ", ???" << 
      abi::__cxa_demangle(typeid(firstArg).name(), 
          nullptr, nullptr, nullptr) << "???";

    ss << printReplayAllocator(std::forward<Args>(args)...);
    return ss.str();
  }

  template <typename... Args>
  static std::string printReplayAllocator(
      int&& firstArg,
      Args&&... args
  )
  {
    std::stringstream ss;

    ss << ", " << firstArg;

    ss << printReplayAllocator(std::forward<Args>(args)...);
    return ss.str();
  }

  template <typename... Args>
  static std::string printReplayAllocator(
      umpire::Allocator&& firstArg,
      Args&&... args
  )
  {
    std::stringstream ss;

    ss << ", rm.getAllocator(\"" << firstArg.getName() << "\")";

    ss << printReplayAllocator(std::forward<Args>(args)...);
    return ss.str();
  }

private:
  Replay(bool enable_replay);
  ~Replay();

  bool replayEnabled;
  static Replay* s_Replay;
};

} /* namespace util */
} /* namespace umpire */

#endif /* UMPIRE_Replay_HPP */
