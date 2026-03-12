//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Macros_HPP
#define UMPIRE_Macros_HPP

#include <cassert>
#include <iostream>
#include <mutex>
#include <sstream>

#include "umpire/config.hpp"
#include "umpire/util/backtrace.hpp"
#include "umpire/util/io.hpp"

#define UMPIRE_ASSERT(condition) assert(condition)

#ifdef UMPIRE_ENABLE_LOGGING
#ifdef UMPIRE_ENABLE_SLIC
#include <stdlib.h>  // for getenv()
#include <strings.h> // for strcasecmp()

#include <string>

#include "slic/GenericOutputStream.hpp"
#include "slic/Logger.hpp"
#define UMPIRE_LOG(lvl, msg)                                                                            \
  {                                                                                                     \
    axom::slic::Logger* plog = axom::slic::Logger::getActiveLogger();                                   \
    if (plog == nullptr) {                                                                              \
      static const std::string env_name = "UMPIRE_LOG_LEVEL";                                           \
      axom::slic::Logger::initialize();                                                                 \
      plog = axom::slic::Logger::getActiveLogger();                                                     \
      axom::slic::message::Level level;                                                                 \
      level = axom::slic::message::Level::Error;                                                        \
      char* enval = std::getenv(env_name.c_str());                                                      \
      if (enval != NULL) {                                                                              \
        for (int i = 0; i < axom::slic::message::Level::Num_Levels; ++i) {                              \
          if (strcasecmp(enval, axom::slic::message::MessageLevelName[i].c_str()) == 0) {               \
            level = (axom::slic::message::Level)i;                                                      \
            break;                                                                                      \
          }                                                                                             \
        }                                                                                               \
      }                                                                                                 \
      plog->setLoggingMsgLevel(level);                                                                  \
                                                                                                        \
      std::string console_format = std::string("[<LEVEL>][<FILE>:<LINE>]: <MESSAGE>\n");                \
      axom::slic::LogStream* console = new axom::slic::GenericOutputStream(&std::cerr, console_format); \
      plog->addStreamToAllMsgLevels(console);                                                           \
    }                                                                                                   \
    std::ostringstream local_msg;                                                                       \
    local_msg << " " << __func__ << " " << msg;                                                         \
    plog->logMessage(axom::slic::message::lvl, local_msg.str(), std::string(__FILE__), __LINE__);       \
  }

#else

#include "umpire/util/Logger.hpp"
#define UMPIRE_LOG(lvl, msg)                                                                           \
  {                                                                                                    \
    if (umpire::util::Logger::getActiveLogger()->logLevelEnabled(umpire::util::message::lvl)) {        \
      std::ostringstream local_msg;                                                                    \
      local_msg << " " << __func__ << " " << msg;                                                      \
      umpire::util::Logger::getActiveLogger()->logMessage(umpire::util::message::lvl, local_msg.str(), \
                                                          std::string(__FILE__), __LINE__);            \
    }                                                                                                  \
  }
#endif // UMPIRE_ENABLE_SLIC

#else

#define UMPIRE_LOG(lvl, msg) ((void)0)

#endif // UMPIRE_ENABLE_LOGGING

#define UMPIRE_UNUSED_ARG(x)

#define UMPIRE_USE_VAR(x) static_cast<void>(x)

#if defined(UMPIRE_ENABLE_BACKTRACE)
#define UMPIRE_RECORD_BACKTRACE(record) \
  umpire::util::backtracer<umpire::util::trace_optional>::get_backtrace(record.allocation_backtrace)
#else
#define UMPIRE_RECORD_BACKTRACE(backtrace) ((void)0)
#endif

#if (__cplusplus >= 201402L)
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated)
#define UMPIRE_HAS_CXX_ATTRIBUTE_DEPRECATED 1
#endif
#endif
#endif

#if defined(UMPIRE_HAS_CXX_ATTRIBUTE_DEPRECATED)
#define UMPIRE_DEPRECATE(Msg) [[deprecated(Msg)]]
#define UMPIRE_DEPRECATE_ALIAS(Msg) [[deprecated(Msg)]]

#elif defined(_MSC_VER)

// for MSVC, use __declspec
#define UMPIRE_DEPRECATE(Msg) __declspec(deprecated(Msg))
#define UMPIRE_DEPRECATE_ALIAS(Msg)

#else

// else use __attribute__(deprecated("Message"))
#define UMPIRE_DEPRECATE(Msg) __attribute__((deprecated(Msg)))
#define UMPIRE_DEPRECATE_ALIAS(Msg)

#endif

#if 0
#define UMPIRE_INTERNAL_TRACK(p, s) registerAllocation(p, s, this);
#define UMPIRE_INTERNAL_UNTRACK(p) deregisterAllocation(p, this);
#else
#define UMPIRE_INTERNAL_TRACK(p, s)
#define UMPIRE_INTERNAL_UNTRACK(p) umpire::util::AllocationRecord{};
#endif

#endif // UMPIRE_Macros_HPP
