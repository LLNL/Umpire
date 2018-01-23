#ifndef UMPIRE_Logger_HPP
#define UMPIRE_Logger_HPP

#include <string>

namespace umpire {
namespace util {

namespace message {
enum Level {
  Error,
  Warning,
  Info,
  Debug,

  Num_Levels
};

static const std::string MessageLevelName[ Level::Num_Levels ] = {
  "ERROR",
  "WARNING",
  "INFO",
  "DEBUG"
};
} /* namespace messge */

class Logger {
  public:

  void setLoggingMsgLevel( message::Level level );

  void logMessage( message::Level level,
                   const std::string& message,
                   const std::string& fileName,
                   int line );

  static void initialize();

  static void finalize();

  static Logger* getActiveLogger();

  static Logger* getRootLogger();

private:
  Logger();
  ~Logger();

  bool m_isEnabled[ message::Num_Levels ];
  static Logger* s_Logger;
};

} /* namespace util */
} /* namespace umpire */

#endif /* UMPIRE_Logger_HPP */
