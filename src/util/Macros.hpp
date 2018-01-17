#ifndef UMPIRE_Macros_HPP
#define UMPIRE_Macros_HPP

#include "umpire/util/Exception.hpp"
#include "umpire/config.hpp"

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
  std::ostringstream local_msg;                                                               \
  local_msg  << " " << __func__ << " " << msg,                                                \
  umpire::util::Logger::getActiveLogger()->logMessage(                                        \
      umpire::util::message::lvl, local_msg.str(),                                            \
      std::string(__FILE__), __LINE__);                                                       \
}
#endif // UMPIRE_ENABLE_SLIC

#else

#define UMPIRE_LOG( lvl, msg )

#endif // UMPIRE_ENABLE_LOGGING

#define UMPIRE_UNUSED_ARG(x)

#endif // UMPIRE_Macros_HPP
