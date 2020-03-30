//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Macros_HPP
#define UMPIRE_Macros_HPP

#include "umpire/util/Backtrace.hpp"
#include "umpire/util/Exception.hpp"
#include "umpire/config.hpp"
#include "umpire/util/io.hpp"

#if defined(UMPIRE_ENABLE_STATISTICS)
#include "umpire/util/statistic_helper.hpp"
#endif

#if defined(UMPIRE_ENABLE_CALIPER)
#include "caliper/cali_datatracker.h"
#endif

#include <sstream>
#include <iostream>
#include <mutex>
#include <cassert>

#define UMPIRE_ASSERT(condition) assert(condition)

#ifdef UMPIRE_ENABLE_LOGGING
#ifdef UMPIRE_ENABLE_SLIC
#include <stdlib.h>   // for getenv()
#include <strings.h>  // for strcasecmp()
#include <string>

#include "slic/Logger.hpp"
#include "slic/GenericOutputStream.hpp"
#define UMPIRE_LOG( lvl, msg )                                                                \
{                                                                                             \
  axom::slic::Logger* plog = axom::slic::Logger::getActiveLogger();                           \
  if ( plog == nullptr ) {                                                                    \
    static const std::string env_name = "UMPIRE_LOG_LEVEL";                                   \
    axom::slic::Logger::initialize();                                                         \
    plog = axom::slic::Logger::getActiveLogger();                                             \
    axom::slic::message::Level level;                                                         \
    level = axom::slic::message::Level::Error;                                                \
    char* enval = getenv(env_name.c_str());                                                   \
    if ( enval != NULL ) {                                                                    \
      for ( int i = 0; i < axom::slic::message::Level::Num_Levels; ++i ) {                    \
        if ( strcasecmp( enval, axom::slic::message::MessageLevelName[ i ].c_str() ) == 0 ) { \
          level = (axom::slic::message::Level)i;                                              \
          break;                                                                              \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
    plog->setLoggingMsgLevel(level);                                                          \
                                                                                              \
    std::string console_format = std::string("[<LEVEL>][<FILE>:<LINE>]: <MESSAGE>\n");        \
    axom::slic::LogStream* console =                                                          \
      new axom::slic::GenericOutputStream( &std::cerr, console_format );                      \
    plog->addStreamToAllMsgLevels( console );                                                 \
                                                                                              \
  }                                                                                           \
  std::ostringstream local_msg;                                                               \
  local_msg  << " " << __func__ << " " << msg;                                                \
  plog->logMessage( axom::slic::message::lvl, local_msg.str(),                                \
                    std::string(__FILE__), __LINE__);                                         \
}

#else

#include "umpire/util/Logger.hpp"
#define UMPIRE_LOG( lvl, msg )                                                                \
{                                                                                             \
  if (umpire::util::Logger::getActiveLogger()->logLevelEnabled(umpire::util::message::lvl)) { \
    std::ostringstream local_msg;                                                             \
    local_msg  << " " << __func__ << " " << msg;                                              \
    umpire::util::Logger::getActiveLogger()->logMessage(                                      \
        umpire::util::message::lvl, local_msg.str(),                                          \
        std::string(__FILE__), __LINE__);                                                     \
  }                                                                                           \
}
#endif // UMPIRE_ENABLE_SLIC

#else

#define UMPIRE_LOG( lvl, msg ) ((void)0)

#endif // UMPIRE_ENABLE_LOGGING

#define UMPIRE_UNUSED_ARG(x)

#define UMPIRE_USE_VAR(x) static_cast<void>(x)

#define UMPIRE_ERROR( msg )                                        \
{                                                                  \
  umpire::util::Backtrace backtrace;                               \
  backtrace.getBacktrace();                                        \
  std::ostringstream umpire_oss_error;                             \
  umpire_oss_error << " " << __func__ << " " << msg << std::endl;  \
  umpire_oss_error << backtrace << std::endl;                      \
  UMPIRE_LOG(Error, umpire_oss_error.str());                       \
  umpire::util::flush_files();                                     \
  throw umpire::util::Exception( umpire_oss_error.str(),           \
                                 std::string(__FILE__),            \
                                 __LINE__);                        \
}

#if defined(UMPIRE_ENABLE_STATISTICS)

#define UMPIRE_RECORD_STATISTIC(name, ...) \
  umpire::util::detail::record_statistic(name, __VA_ARGS__);

#else

#define UMPIRE_RECORD_STATISTIC(name, ...) ((void) 0)

#endif // defined(UMPIRE_ENABLE_STATISTICS)

#if defined(UMPIRE_ENABLE_CALIPER)

#define UMPIRE_CALIPER_TRACK(ptr, name, size) cali_datatracker_track(ptr, name, size)
#define UMPIRE_CALIPER_UNTRACK(ptr) cali_datatracker_untrack(ptr)

#else

#define UMPIRE_CALIPER_TRACK(ptr, name, size)
#define UMPIRE_CALIPER_UNTRACK(ptr)

#endif

#endif // UMPIRE_Macros_HPP
