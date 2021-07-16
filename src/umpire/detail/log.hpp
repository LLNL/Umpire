#pragma once

#include "umpire/exception.hpp"
#include "umpire/detail/logger.hpp"

#include <sstream>

#define UMPIRE_LOG( lvl, msg )                                                                \
{                                                                                             \
  if (umpire::detail::logger::getActiveLogger()->logLevelEnabled(umpire::detail::message::lvl)) { \
    std::ostringstream local_msg;                                                             \
    local_msg  << " " << __func__ << " " << msg;                                              \
    umpire::detail::logger::getActiveLogger()->logMessage(                                      \
        umpire::detail::message::lvl, local_msg.str(),                                          \
        std::string(__FILE__), __LINE__);                                                     \
  }                                                                                           \
}

#define UMPIRE_ERROR(msg) \
{                                                                                                   \
  std::ostringstream umpire_oss_error;                                                              \
  umpire_oss_error << " " << __func__ << " " << msg << std::endl;                                   \
  throw umpire::exception(umpire_oss_error.str(),                                            \
                          std::string(__FILE__),                                             \
                          __LINE__);                                                         \
}
