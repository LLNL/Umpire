//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/io.hpp"

#include <stdlib.h> // for getenv()

#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include "umpire/config.hpp"
#include "umpire/util/MPI.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/OutputBuffer.hpp"

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

std::ostream& replay()
{
  static std::ostream out{nullptr};
  return out;
}

std::ostream& error()
{
  static std::ostream out{std::cerr.rdbuf()};
  return out;
}

namespace util {

static std::string make_unique_filename(const std::string& base_dir,
                                        const std::string& name, const int pid,
                                        const std::string& extension);

static inline bool file_exists(const std::string& file);

static inline bool directory_exists(const std::string& file);

void initialize_io(const bool enable_log, const bool enable_replay)
{
  static util::OutputBuffer s_log_buffer;
  static util::OutputBuffer s_replay_buffer;
  static util::OutputBuffer s_error_buffer;

  s_log_buffer.setConsoleStream(nullptr);
  s_replay_buffer.setConsoleStream(nullptr);
  s_error_buffer.setConsoleStream(&std::cerr);

  log().rdbuf(&s_log_buffer);
  replay().rdbuf(&s_replay_buffer);
  error().rdbuf(&s_error_buffer);

  std::string root_io_dir{"./"};
  const char* output_dir{std::getenv("UMPIRE_OUTPUT_DIR")};
  if (output_dir)
    root_io_dir = output_dir;

  std::string file_basename{"umpire"};
  const char* base_name{std::getenv("UMPIRE_OUTPUT_BASENAME")};
  if (base_name)
    file_basename = base_name;

  const int pid{getpid()};

  const std::string log_filename{
      make_unique_filename(root_io_dir, file_basename, pid, "log")};

  const std::string replay_filename{
      make_unique_filename(root_io_dir, file_basename, pid, "replay")};

  const std::string error_filename{
      make_unique_filename(root_io_dir, file_basename, pid, "error")};

  if (!directory_exists(root_io_dir)) {
    if (MPI::isInitialized()) {
      if (MPI::getRank() == 0) {
#if defined(UMPIRE_ENABLE_FILESYSTEM)
        std::filesystem::path root_io_dir_path{root_io_dir};

        if (!std::filesystem::exists(root_io_dir_path) &&
            (enable_log || enable_replay)) {
          std::filesystem::create_directories(root_io_dir_path);
        }
#else
        struct stat info;
        if (stat(root_io_dir.c_str(), &info)) {
          if (enable_log || enable_replay) {
#ifndef WIN32
            if (mkdir(root_io_dir.c_str(),
                      S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
              UMPIRE_ERROR("mkdir(" << root_io_dir << ") failed");
            }
#else
            if (_mkdir(root_io_dir.c_str())) {
              UMPIRE_ERROR("mkdir(" << root_io_dir << ") failed");
            }
#endif
          }
        } else if (!(S_ISDIR(info.st_mode))) {
          UMPIRE_ERROR(root_io_dir << "exists and is not a directory");
        }
#endif
      }
      MPI::sync();
    } else {
      UMPIRE_ERROR(
          "Cannot create output directory before MPI has been initialized. "
          "Please unset UMPIRE_OUTPUT_DIR in your environment");
    }
  }

  if (enable_log) {
    static std::ofstream s_log_ofstream{log_filename};

    if (s_log_ofstream) {
      s_log_buffer.setFileStream(&s_log_ofstream);
    } else {
      UMPIRE_ERROR("Couldn't open log file:" << log_filename);
    }
  }

  if (enable_replay) {
    static std::ofstream s_replay_ofstream{replay_filename};

    if (s_replay_ofstream) {
      s_replay_buffer.setFileStream(&s_replay_ofstream);
    } else {
      UMPIRE_ERROR("Couldn't open replay file:" << replay_filename);
    }
  }

  MPI::logMpiInfo();
}

void flush_files()
{
  log().flush();
  replay().flush();
  error().flush();
}

static std::string make_unique_filename(const std::string& base_dir,
                                        const std::string& name, const int pid,
                                        const std::string& extension)
{
  int unique_id{0};
  std::string filename;

  do {
    filename = base_dir + "/" + name + "." + std::to_string(pid) + "." +
               std::to_string(unique_id++) + "." + extension;
  } while (file_exists(filename));

  return filename;
}

static inline bool file_exists(const std::string& path)
{
  std::ifstream ifile(path.c_str());
  return ifile.good();
}

static inline bool directory_exists(const std::string& path)
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

} // end namespace util
} // end namespace umpire
