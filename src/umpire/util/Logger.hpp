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
#ifndef UMPIRE_Logger_HPP
#define UMPIRE_Logger_HPP

#include <string>

namespace umpire {
namespace util {

namespace message {
enum Level {
  Error,
  Warning,
  Info,
  Debug,

  Num_Levels
};

static const std::string MessageLevelName[ Level::Num_Levels ] = {
  "ERROR",
  "WARNING",
  "INFO",
  "DEBUG"
};
} /* namespace messge */

class Logger {
  public:

  void setLoggingMsgLevel( message::Level level ) noexcept;

  void logMessage( message::Level level,
                   const std::string& message,
                   const std::string& fileName,
                   int line ) noexcept;

  static void initialize();

  static void finalize();

  static Logger* getActiveLogger();

  inline bool logLevelEnabled( message::Level level )
  {
    if ( level < 0 || level >= message::Num_Levels || m_isEnabled[ level ] == false  )
      return false;
    else
      return true;
  };

  ~Logger() noexcept = default;
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

private:
  Logger() noexcept;

  bool m_isEnabled[ message::Num_Levels ];
};

} /* namespace util */
} /* namespace umpire */

#endif /* UMPIRE_Logger_HPP */
