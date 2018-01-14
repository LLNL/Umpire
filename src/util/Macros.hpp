#ifndef UMPIRE_Macros_HPP
#define UMPIRE_Macros_HPP

#include "umpire/util/Exception.hpp"

#include <sstream>
#include <iostream>

#define UMPIRE_ERROR( msg )                                        \
{                                                                  \
  std::ostringstream umpire_oss_error;                             \
  umpire_oss_error << msg;                                         \
  throw umpire::util::Exception( umpire_oss_error.str(),           \
                                 std::string(__FILE__),            \
                                 __LINE__);                        \
}

#if defined(NDEBUG)

#define UMPIRE_LOG( lvl, msg )

#else
#include <stdlib.h>
#include <strings.h>

#if defined(USE_SLIC_FOR_UMPIRE_LOG)
#include "slic/Logger.hpp"
#define UMPIRE_LOG( lvl, msg )                                                                \
{                                                                                             \
  axom::slic::Logger* plog = axom::slic::Logger::getActiveLogger();                           \
  if ( plog == nullptr ) {                                                                    \
    axom::slic::Logger::initialize();                                                         \
    plog = axom::slic::Logger::getActiveLogger();                                             \
    axom::slic::message::Level level;                                                         \
    level = axom::slic::message::Level::Error;                                                \
    char* enval = getenv("UMPIRE_LOG_LEVEL");                                               \
    if ( enval != NULL ) {                                                                    \
      for ( int i = 0; i < axom::slic::message::Level::Num_Levels; ++i ) {                    \
        if ( strcasecmp( enval, axom::slic::message::MessageLevelName[ i ].c_str() ) == 0 ) { \
          level = (axom::slic::message::Level)i;                                              \
          break;                                                                              \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
    plog->setLoggingMsgLevel(level);                                                          \
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
  std::ostringstream local_msg;                                                               \
  local_msg  << " " << __func__ << " " << msg,                                                \
  umpire::util::Logger::getActiveLogger()->logMessage(                                        \
      umpire::util::message::lvl, local_msg.str(),                                            \
      std::string(__FILE__), __LINE__);                                                       \
}
#endif // defined(USE_SLIC_FOR_UMPIRE_LOG)

#endif // defined(NDEBUG)

#define UMPIRE_UNUSED_ARG(x)

#endif // UMPIRE_Macros_HPP
