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
#ifndef UMPIRE_Macros_HPP
#define UMPIRE_Macros_HPP

#include "umpire/util/Exception.hpp"
#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_STATISTICS)
#include "umpire/util/statistic_helper.hpp"
#endif

#include <sstream>
#include <iostream>
#include <mutex>

#ifdef UMPIRE_ENABLE_ASSERTS
#include <cassert>
#define UMPIRE_ASSERT(condition) assert(condition)
#else
#define UMPIRE_ASSERT(condition) ((void)0)
#endif // UMPIRE_ENABLE_ASSERTS

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
  UMPIRE_LOG(Error, msg);                                          \
  std::ostringstream umpire_oss_error;                             \
  umpire_oss_error << " " << __func__ << " " << msg;               \
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

#define UMPIRE_LOCK \
  if ( !m_mutex->try_lock() ) \
    m_mutex->lock();

#define UMPIRE_UNLOCK \
  m_mutex->unlock();

#define UMPIRE_CHECK_ALLOCATOR(record, name) \
  if (record.strategy != this) { \
    UMPIRE_ERROR(ptr << " was not allocated by " << name); \
  } \

#endif // UMPIRE_Macros_HPP
