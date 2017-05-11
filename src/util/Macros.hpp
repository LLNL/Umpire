#ifndef UMPIRE_Macros_HPP
#define UMPIRE_Macros_HPP

#include "umpire/util/Exception.hpp"

#include <sstream>

#define UMPIRE_ERROR( msg )                                        \
{                                                                  \
  std::ostringstream umpire_oss_error;                             \
  umpire_oss_error << msg;                                         \
  throw umpire::util::Exception( umpire_oss_error.str(),           \
                                 std::string(__FILE__),            \
                                 __LINE__);                        \
}

#define UMPIRE_LOG( msg )                                          \
{                                                                  \
  std::ostringstream umpire_oss_log;                               \
  umpire_oss_log << msg;                                           \
  std::cout << "[" << __FILE__  << ":" << __LINE__ << "]:";        \
  std::cout << msg << std::endl;                                   \
}

#endif // UMPIRE_Macros_HPP
