//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/io.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include "umpire/config.hpp"
#include "umpire/util/MPI.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/OutputBuffer.hpp"
#include "umpire/util/error.hpp"

#if defined(UMPIRE_ENABLE_FILESYSTEM)
#include <filesystem>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#if !defined(_MSC_VER)
#include <unistd.h> // getpid()
#else
#include <process.h>
#define getpid _getpid
#include <direct.h>
#endif

namespace umpire {

std::ostream& log()
{
  static std::ostream out{std::cout.rdbuf()};
  return out;
}

std::ostream& error()
{
  static std::ostream out{std::cerr.rdbuf()};
  return out;
}

namespace util {

namespace detail {

OutputBuffer& s_log_buffer_accessor()
{
  static OutputBuffer buffer;
  return buffer;
}

OutputBuffer& s_error_buffer_accessor()
{
  static OutputBuffer buffer;
  return buffer;
}

} // namespace detail

void initialize_io(const bool enable_log)
{
  OutputBuffer& s_log_buffer = detail::s_log_buffer_accessor();
  OutputBuffer& s_error_buffer = detail::s_error_buffer_accessor();

  s_log_buffer.setConsoleStream(nullptr);
  s_error_buffer.setConsoleStream(&std::cerr);

  log().rdbuf(&s_log_buffer);
  error().rdbuf(&s_error_buffer);

  const std::string& root_io_dir{util::get_io_output_dir()};
  const std::string& file_basename{util::get_io_output_basename()};

  const int pid{getpid()};

  const std::string log_filename{make_unique_filename(root_io_dir, file_basename, pid, "log")};

  const std::string error_filename{make_unique_filename(root_io_dir, file_basename, pid, "error")};

  if (!directory_exists(root_io_dir)) {
    if (MPI::isInitialized()) {
      if (MPI::getRank() == 0) {
#if defined(UMPIRE_ENABLE_FILESYSTEM)
        std::filesystem::path root_io_dir_path{root_io_dir};

        if (!std::filesystem::exists(root_io_dir_path) && enable_log) {
          std::filesystem::create_directories(root_io_dir_path);
        }
#else
        struct stat info;
        if (stat(root_io_dir.c_str(), &info)) {
          if (enable_log) {
#ifndef WIN32
            if (mkdir(root_io_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
              UMPIRE_ERROR(runtime_error, fmt::format("mkdir({}) failed", root_io_dir));
            }
#else
            if (_mkdir(root_io_dir.c_str())) {
              UMPIRE_ERROR(runtime_error, fmt::format("mkdir( \"{}\" ) failed", root_io_dir));
            }
#endif
          }
        } else if (!(S_ISDIR(info.st_mode))) {
          UMPIRE_ERROR(runtime_error, fmt::format("{} exists and is not a directory", root_io_dir));
        }
#endif
      }
      MPI::sync();
    } else {
      UMPIRE_ERROR(runtime_error,
                   "Cannot create output directory before MPI has been initialized. Please unset UMPIRE_OUTPUT_DIR in "
                   "your environment");
    }
  }

  if (enable_log) {
    static std::ofstream s_log_ofstream{log_filename};

    if (s_log_ofstream) {
      s_log_buffer.setFileStream(&s_log_ofstream);
    } else {
      UMPIRE_ERROR(runtime_error, fmt::format("Couldn't open log file: {}", log_filename));
    }
  }

  MPI::logMpiInfo();
}

void finalize_io()
{
  detail::s_log_buffer_accessor().sync();
  detail::s_log_buffer_accessor().setConsoleStream(nullptr);
  detail::s_log_buffer_accessor().setFileStream(nullptr);
  detail::s_error_buffer_accessor().sync();
  detail::s_error_buffer_accessor().setConsoleStream(nullptr);
  detail::s_error_buffer_accessor().setFileStream(nullptr);
}

void flush_files()
{
  log().flush();
  error().flush();
}

std::string make_unique_filename(const std::string& base_dir, const std::string& name, const int pid,
                                 const std::string& extension)
{
  int unique_id{0};
  std::string filename;

  do {
    filename = base_dir + "/" + name + "." + std::to_string(pid) + "." + std::to_string(unique_id++) + "." + extension;
  } while (file_exists(filename));

  return filename;
}

bool file_exists(const std::string& path)
{
  std::ifstream ifile(path.c_str());
  return ifile.good();
}

bool directory_exists(const std::string& path)
{
#if defined(UMPIRE_ENABLE_FILESYSTEM)
  std::filesystem::path fspath_path(path);
  return std::filesystem::exists(fspath_path);
#else
  struct stat info;
  if (stat(path.c_str(), &info)) {
    return false;
  } else {
    return S_ISDIR(info.st_mode);
  }
#endif
}

const std::string& get_io_output_dir()
{
  static const char* output_dir_env{std::getenv("UMPIRE_OUTPUT_DIR")};
  static const std::string output_dir = output_dir_env ? output_dir_env : "./";

  return output_dir;
}

const std::string& get_io_output_basename()
{
  static const char* base_name_env{std::getenv("UMPIRE_OUTPUT_BASENAME")};
  static std::string base_name = base_name_env ? base_name_env : "umpire";

  return base_name;
}

} // end namespace util
} // end namespace umpire
