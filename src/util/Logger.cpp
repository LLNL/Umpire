#include "umpire/util/Logger.hpp"

#include <iostream>   // for std::cout, std::cerr

namespace umpire {
namespace util {

Logger* Logger::s_Logger = nullptr;

Logger::Logger()
{
  // by default, all message streams are disabled
  for ( int i=0 ; i < message::Num_Levels ; ++i )
    m_isEnabled[ i ] = false;
}

Logger::~Logger()
{
}

void Logger::setLoggingMsgLevel( message::Level level )
{
  for ( int i=0 ; i < message::Num_Levels ; ++i )
    m_isEnabled[ i ] = (i<= level) ? true : false;
}

void Logger::logMessage( message::Level level,
                         const std::string& message,
                         const std::string& fileName,
                         int line )
{
  if ( level < 0 || level >= message::Num_Levels || m_isEnabled[ level ] == false  )
    return;   /* short-circuit */

  std::cout 
    << "[" << message::MessageLevelName[ level ] << "]"
    << "[" << fileName  << ":" << line << "]:" 
    << message 
    << std::endl;
}

void Logger::initialize()
{
  if ( s_Logger != nullptr )
    return;

  s_Logger = new Logger();
}

void Logger::finalize()
{
  delete s_Logger;
  s_Logger = nullptr;
}

Logger* Logger::getActiveLogger()
{
  return s_Logger;
}

} /* namespace util */
} /* namespace umpire */
