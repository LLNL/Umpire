//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/Logger.hpp"

#include <algorithm> // for std::equal
#include <cctype>    // for std::toupper
#include <cstdlib>   // for getenv()

#include "umpire/util/io.hpp"

namespace umpire {
namespace util {

static const char* env_name = "UMPIRE_LOG_LEVEL";
static message::Level defaultLevel = message::Info;

static const char* MessageLevelName[message::Num_Levels] = {"ERROR", "WARNING",
                                                            "INFO", "DEBUG"};

static int case_insensitive_match(const std::string s1, const std::string s2)
{
  return (s1.size() == s2.size()) &&
         std::equal(s1.begin(), s1.end(), s2.begin(), [](char c1, char c2) {
           return (std::toupper(c1) == std::toupper(c2));
         });
}

Logger::Logger() noexcept
    : // by default, all message streams are disabled
      m_is_enabled{false, false, false, false}
{
  message::Level level{defaultLevel};
  const char* enval = getenv(env_name);

  if (enval) {
    for (int i = 0; i < message::Num_Levels; ++i) {
      if (case_insensitive_match(enval, MessageLevelName[i])) {
        level = static_cast<message::Level>(i);
        break;
      }
    }
  }

  setLoggingMsgLevel(level);
}

void Logger::setLoggingMsgLevel(message::Level level) noexcept
{
  for (int i = 0; i < message::Num_Levels; ++i)
    m_is_enabled[i] = (i <= level);
}

void Logger::logMessage(message::Level level, const std::string& message,
                        const std::string& fileName, int line) noexcept
{
  if (!logLevelEnabled(level))
    return;

  umpire::log() << "[" << MessageLevelName[level] << "]"
                << "[" << fileName << ":" << line << "]:" << message
                << std::endl;
}

Logger* Logger::getActiveLogger()
{
  static Logger logger;
  return &logger;
}

} // end namespace util
} // end namespace umpire
