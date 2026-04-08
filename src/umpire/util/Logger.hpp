//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Logger_HPP
#define UMPIRE_Logger_HPP

#include <memory>
#include <string>

// Forward declare spdlog types to avoid including spdlog.h in header
namespace spdlog {
class logger;
namespace level {
enum level_enum : int;
}
} // namespace spdlog

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
} // end namespace message

class Logger {
 public:
  // Initialize the global logger (called once from ResourceManager)
  static void initialize();

  // Finalize and flush logs (called from ResourceManager destructor)
  static void finalize();

  // Check if a log level should be logged (for macro optimization)
  static bool shouldLog(message::Level level) noexcept;

  // Log a message at the specified level
  static void log(message::Level level, const std::string& message,
                  const std::string& fileName, int line) noexcept;

  // Get the underlying spdlog logger (for advanced usage)
  static std::shared_ptr<spdlog::logger> getSpdlogger();

 private:
  Logger() = delete;
  ~Logger() = delete;

  static spdlog::level::level_enum convertLevel(message::Level level) noexcept;
  static message::Level parseEnvLogLevel() noexcept;
  static std::string generateLogFilename();

  static std::shared_ptr<spdlog::logger> s_logger;
  static message::Level s_level;
  static bool s_initialized;
};

} // end namespace util
} // end namespace umpire

#endif // UMPIRE_Logger_HPP
