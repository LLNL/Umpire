//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/config.hpp"
#include "umpire/util/IOManager.hpp"
#include "umpire/util/OutputBuffer.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/MPI.hpp"

#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>   // for getenv()

#if defined(UMPIRE_ENABLE_FILESYSTEM)
#include <filesystem>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif

#if !defined(_MSC_VER)
#include <unistd.h>   // getpid()
#else
#include <process.h>
#define getpid _getpid
#endif

namespace umpire {

static util::OutputBuffer log_buffer;
static util::OutputBuffer replay_buffer;
static util::OutputBuffer error_buffer;

std::ostream log(&log_buffer);
std::ostream replay(&replay_buffer);
std::ostream error(&error_buffer);

namespace util {

std::string IOManager::s_root_io_dir;
std::string IOManager::s_file_basename;

std::string IOManager::s_log_filename;
std::string IOManager::s_replay_filename;
std::string IOManager::s_error_filename;

std::ofstream* IOManager::s_log_ofstream;
std::ofstream* IOManager::s_replay_ofstream;

bool IOManager::s_initialized = false;

static std::string makeUniqueFilename(
  const std::string& base_dir,
  const std::string& name,
  int rank,
  int pid,
  const std::string& extension);

static inline bool fileExists(const std::string& file);


void
IOManager::setOutputDir(const std::string& dir)
{
  if (s_initialized) {
    UMPIRE_ERROR("Can't change IOManager output directory once initialized!");
  }

  s_root_io_dir = dir;
}

void
IOManager::initialize(
    bool enable_log,
    bool enable_replay)
{
  if (!s_initialized) {
    s_root_io_dir = "./";
    s_file_basename = "umpire";

    auto output_dir = std::getenv("UMPIRE_OUTPUT_DIR");

    if (output_dir) {
      s_root_io_dir = std::string(output_dir);
    }

    auto base_name = std::getenv("UMPIRE_OUTPUT_BASENAME");

    if (base_name) {
      s_file_basename = std::string(base_name);
    }

    auto rank = MPI::getRank();
    int pid{getpid()};

    s_log_filename = makeUniqueFilename(s_root_io_dir, s_file_basename, rank, pid, "log");
    s_replay_filename = makeUniqueFilename(s_root_io_dir, s_file_basename, rank, pid, "replay");
    s_error_filename = makeUniqueFilename(s_root_io_dir, s_file_basename, rank, pid, "error");
    s_error_filename = "";

    log_buffer.setConsoleStream(&std::cout);
    replay_buffer.setConsoleStream(nullptr);
    error_buffer.setConsoleStream(&std::cerr);

    if (rank == 0) {
#if defined(UMPIRE_ENABLE_FILESYSTEM)
      std::filesystem::path root_io_dir_path(s_root_io_dir);

      if (!std::filesystem::exists(root_io_dir_path))
      {
        if (enable_log || enable_replay) {
          std::filesystem::create_directories(root_io_dir_path);
        }
      }
#else
      struct stat info;
      if ( stat( s_root_io_dir.c_str(), &info ) )
      {
        if (enable_log || enable_replay) {
          if ( mkdir(s_root_io_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) )
          {
            UMPIRE_ERROR("mkdir(" << s_root_io_dir << ") failed");
          }
        }
      }
      else if ( !(S_ISDIR(info.st_mode)) )
      {
        UMPIRE_ERROR(s_root_io_dir << "exists and is not a directory");
      }
#endif
    }

    if (enable_log) {
      s_log_ofstream = new std::ofstream(s_log_filename);

      if (*s_log_ofstream) {
        log_buffer.setFileStream(s_log_ofstream);
      } else {
        UMPIRE_ERROR("Couldn't open log file:" << s_log_filename);
      }
    }

    if (enable_replay) {
      s_replay_ofstream = new std::ofstream(s_replay_filename);

      if (*s_replay_ofstream) {
        replay_buffer.setFileStream(s_replay_ofstream);
      } else {
        UMPIRE_ERROR("Couldn't open replay file:" << s_log_filename);
      }
    }

    s_initialized = true;
  }
}

static std::string
makeUniqueFilename(
    const std::string& base_dir,
    const std::string& name,
    int rank,
    int pid,
    const std::string& extension)
{
  int unique_id = -1;
  std::stringstream ss;
  std::string filename;

  do {
    ss.str("");
    ss.clear();
    unique_id++;
    ss << base_dir << "/" << name << "." << rank << "." << pid << "." << unique_id << "." << extension;
    filename = ss.str();
  } while (fileExists(filename));

  return filename;
}

static inline bool fileExists(const std::string& path)
{
  std::ifstream ifile(path.c_str());
  return ifile.good();
}

} // end of namespace util
} // end of namespace umpire
