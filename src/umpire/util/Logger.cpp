//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/Logger.hpp"
#include "umpire/util/io.hpp"
#include "umpire/util/string_utils.hpp"

#include <cstdlib>    // for getenv()
#include <cctype>     // for std::toupper
#include <algorithm>  // for std::equal

#include "umpire/tpl/json/json.hpp"

namespace umpire {
namespace util {

static const char* MessageLevelName[message::Num_Levels] = {
  "ERROR",
  "WARNING",
  "INFO",
  "DEBUG"
};

Logger::Logger() noexcept :
  // by default, all message streams are disabled
  m_is_enabled{false, false, false, false}
{
  static const char* env_name = "UMPIRE_LOG_LEVEL";
  static const char* config_env_var = "UMPIRE_LOG_CFG";

  static message::Level defaultLevel = message::Info;
  message::Level level{defaultLevel};

  const char* env = getenv(env_name);
  if (env) {
    for (int i = 0; i < message::Num_Levels; ++i) {
      if (case_insensitive_match(env, MessageLevelName[i])) {
        level = static_cast<message::Level>(i);
        break;
      }
    }
  }

  const char* config_env = std::getenv(config_env_var);
  if (config_env) {
    auto json = nlohmann::json::parse(std::string{config_env});

    auto level_from_cfg = json["level"];
    for (int i = 0; i < message::Num_Levels; ++i) {
      if (case_insensitive_match(level_from_cfg, MessageLevelName[i])) {
        level = static_cast<message::Level>(i);
        break;
      }
    }
  }

  setLoggingMsgLevel(level);
}

void Logger::setLoggingMsgLevel( message::Level level ) noexcept
{
  for (int i=0; i < message::Num_Levels; ++i)
    m_is_enabled[i] = (i<=level);
}

void Logger::logMessage( message::Level level,
                         const std::string& message,
                         const std::string& fileName,
                         int line ) noexcept
{
  if (!logLevelEnabled(level)) return;

  umpire::log()
    << "[" << MessageLevelName[level] << "]"
    << "[" << fileName  << ":" << line << "]:"
    << message
    << std::endl;
}

Logger* Logger::getActiveLogger()
{
  static Logger logger;
  return &logger;
}

} // end namespace util
} // end namespace umpire
