//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/Logger.hpp"
#include "umpire/util/IOManager.hpp"

#if !defined(_MSC_VER)
#include <strings.h>  // for strcasecmp()
#else
#define strcasecmp _stricmp
#endif

#include <iostream>   // for std::cout, std::cerr
#include <stdlib.h>   // for getenv()

namespace umpire {
namespace util {

static const char* env_name = "UMPIRE_LOG_LEVEL";
static message::Level defaultLevel = message::Info;

static const std::string MessageLevelName[ message::Num_Levels ] = {
  "ERROR",
  "WARNING",
  "INFO",
  "DEBUG"
};

Logger::Logger() noexcept
{
  // by default, all message streams are disabled
  for ( int i=0 ; i < message::Num_Levels ; ++i )
    m_isEnabled[ i ] = false;

  message::Level level = defaultLevel;
  char* enval = getenv(env_name);

  if ( enval != NULL ) {
    for ( int i = 0; i < message::Num_Levels; ++i ) {
      if ( strcasecmp( enval, MessageLevelName[ i ].c_str() ) == 0 ) {
        level = (message::Level)i;
        break;
      }
    }
  }

  setLoggingMsgLevel(level);
}

void Logger::setLoggingMsgLevel( message::Level level ) noexcept
{
  for ( int i=0 ; i < message::Num_Levels ; ++i )
    m_isEnabled[ i ] = (i<= level) ? true : false;
}

void Logger::logMessage( message::Level level,
                         const std::string& message,
                         const std::string& fileName,
                         int line ) noexcept
{
  if ( !logLevelEnabled( level ) )
    return;   /* short-circuit */

  umpire::log
    << "[" << MessageLevelName[ level ] << "]"
    << "[" << fileName  << ":" << line << "]:"
    << message
    << std::endl;
}

Logger* Logger::getActiveLogger()
{
  static Logger logger;

  return &logger;
}

} /* namespace util */
} /* namespace umpire */
