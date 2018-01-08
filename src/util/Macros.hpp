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

#include "umpire/util/Logger.hpp"

#define UMPIRE_LOG( lvl, msg )                                     \
{                                                                  \
  std::ostringstream local_msg;                                    \
  local_msg << msg;                                                \
  umpire::util::Logger::getActiveLogger()->logMessage(             \
      umpire::util::message::lvl,                                  \
      local_msg.str(),                                             \
      std::string(__FILE__),                                       \
      __LINE__);                                                   \
}

#endif // defined(NDEBUG)

#define UMPIRE_UNUSED_ARG(x)

#endif // UMPIRE_Macros_HPP
