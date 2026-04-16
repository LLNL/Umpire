//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/util/Logger.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <vector>

#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/null_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include "umpire/util/io.hpp"

#if !defined(_MSC_VER)
#include <unistd.h>
#else
#include <process.h>
#define getpid _getpid
#endif

namespace umpire {
namespace util {

// Static member initialization
std::shared_ptr<spdlog::logger> Logger::s_logger = nullptr;
message::Level Logger::s_level = message::Info;
bool Logger::s_initialized = false;

static const char* MessageLevelName[message::Num_Levels] = {"ERROR", "WARNING", "INFO", "DEBUG"};

static bool case_insensitive_match(const std::string& s1, const std::string& s2)
{
  return (s1.size() == s2.size()) &&
         std::equal(s1.begin(), s1.end(), s2.begin(),
                    [](char c1, char c2) { return std::toupper(c1) == std::toupper(c2); });
}

message::Level Logger::parseEnvLogLevel() noexcept
{
  const char* env_level = std::getenv("UMPIRE_LOG_LEVEL");
  if (!env_level) {
    return message::Info; // default
  }

  std::string level_str(env_level);
  for (int i = 0; i < message::Num_Levels; ++i) {
    if (case_insensitive_match(level_str, MessageLevelName[i])) {
      return static_cast<message::Level>(i);
    }
  }

  return message::Info; // fallback
}

std::string Logger::generateLogFilename()
{
  const std::string& output_dir = get_io_output_dir();
  const std::string& basename = get_io_output_basename();
  const int pid = getpid();

  // Reuse existing make_unique_filename from io.cpp
  return make_unique_filename(output_dir, basename, pid, "log");
}

spdlog::level::level_enum Logger::convertLevel(message::Level level) noexcept
{
  switch (level) {
  case message::Error:
    return spdlog::level::err;
  case message::Warning:
    return spdlog::level::warn;
  case message::Info:
    return spdlog::level::info;
  case message::Debug:
    return spdlog::level::debug;
  default:
    return spdlog::level::info;
  }
}

void Logger::initialize()
{
  if (s_initialized) {
    return;
  }

  // Parse log level from environment
  s_level = parseEnvLogLevel();

  // Check if logging is enabled (UMPIRE_LOG_LEVEL must be set)
  const char* env_enable_log = std::getenv("UMPIRE_LOG_LEVEL");
  if (!env_enable_log) {
    // Logging disabled - create a null logger
    // Don't use spdlog::register_logger to avoid static initialization order issues
    auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    s_logger = std::make_shared<spdlog::logger>("umpire", null_sink);
    s_initialized = true;
    return;
  }

  // Check for async mode
  const char* env_async = std::getenv("UMPIRE_LOG_ASYNC");
  const bool use_async = (env_async && (std::string(env_async) == "1" ||
                                        case_insensitive_match(env_async, "true") ||
                                        case_insensitive_match(env_async, "on")));

  // Get queue size for async logging
  const char* env_queue_size = std::getenv("UMPIRE_LOG_QUEUE_SIZE");
  const size_t queue_size = env_queue_size ? std::atoi(env_queue_size) : 8192;

  // Initialize async thread pool if needed
  if (use_async && !spdlog::thread_pool()) {
    spdlog::init_thread_pool(queue_size, 1);
  }

  // Create sinks
  std::vector<spdlog::sink_ptr> sinks;

  // File sink (always enabled when UMPIRE_LOG_LEVEL is set)
  std::string log_filename = generateLogFilename();
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_filename, false);
  sinks.push_back(file_sink);

  // Optional console sink
  const char* env_console = std::getenv("UMPIRE_LOG_TO_CONSOLE");
  const bool log_to_console = (env_console && (std::string(env_console) == "1" ||
                                                case_insensitive_match(env_console, "true") ||
                                                case_insensitive_match(env_console, "on")));
  if (log_to_console) {
    auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    sinks.push_back(console_sink);
  }

  // Create logger (sync or async)
  // Don't use spdlog::register_logger to avoid static initialization order issues
  if (use_async) {
    s_logger = std::make_shared<spdlog::async_logger>("umpire", sinks.begin(), sinks.end(),
                                                       spdlog::thread_pool(),
                                                       spdlog::async_overflow_policy::block);
  } else {
    s_logger = std::make_shared<spdlog::logger>("umpire", sinks.begin(), sinks.end());
  }

  // Set format pattern to match current output: [LEVEL][file:line]: message
  s_logger->set_pattern("[%^%L%$][%s:%#]: %v");

  // Set log level
  s_logger->set_level(convertLevel(s_level));

  // Flush on every message for Error level (safety)
  s_logger->flush_on(spdlog::level::err);

  s_initialized = true;
}

void Logger::finalize()
{
  // Do nothing during finalize to avoid static destruction order issues.
  // When ResourceManager is destroyed during static destruction (e.g., in tests
  // that create a static ResourceManager reference), attempting to clean up the
  // logger can cause segfaults because:
  // 1. spdlog's internal state (sinks, file handles) may already be destroyed
  // 2. Even resetting the shared_ptr can crash if the control block is corrupted
  //
  // It's safe to leak the logger here because:
  // - Normal program exit will close all file handles and free all memory
  // - The logger's file sink will flush on destruction if still valid
  // - Process cleanup handles everything automatically
  //
  // This is only an issue for tests with static ResourceManager instances.
  // Normal usage (where ResourceManager is created/destroyed in main) works fine.
}

bool Logger::shouldLog(message::Level level) noexcept
{
  if (!s_initialized || !s_logger) {
    return false;
  }
  return level <= s_level;
}

void Logger::log(message::Level level, const std::string& message, const std::string& fileName,
                 int line) noexcept
{
  if (!s_initialized || !s_logger || !shouldLog(level)) {
    return;
  }

  // Use spdlog's source location logging
  s_logger->log(spdlog::source_loc{fileName.c_str(), line, ""}, convertLevel(level), message);
}

std::shared_ptr<spdlog::logger> Logger::getSpdlogger()
{
  return s_logger;
}

} // end namespace util
} // end namespace umpire
