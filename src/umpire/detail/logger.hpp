//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Logger_HPP
#define UMPIRE_Logger_HPP

#include <string>

namespace umpire {
namespace detail {

namespace message {
enum Level {
  Error,
  Warning,
  Info,
  Debug,

  Num_Levels
};
} // end namespace message

class logger {
  public:

  void setLoggingMsgLevel(message::Level level) noexcept;

  void logMessage(message::Level level,
                  const std::string& message,
                  const std::string& fileName,
                  int line) noexcept;

  static void initialize();

  static void finalize();

  static logger* getActiveLogger();

  inline bool logLevelEnabled(message::Level level)
  {
    if ( level < 0 || level >= message::Num_Levels || m_is_enabled[ level ] == false  )
      return false;
    else
      return true;
  };

  ~logger() noexcept = default;
  logger(const logger&) = delete;
  logger& operator=(const logger&) = delete;

private:
  logger() noexcept;

  bool m_is_enabled[message::Num_Levels];
};

} // end namespace util
} // end namespace umpire

#endif // UMPIRE_Logger_HPP
