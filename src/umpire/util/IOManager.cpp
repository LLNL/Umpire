//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/IOManager.hpp"
#include "umpire/util/OutputBuffer.hpp"
#include "umpire/util/Macros.hpp"

#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>

#if !defined(_MSC_VER)
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif

namespace umpire {

static util::OutputBuffer log_buffer;
static util::OutputBuffer replay_buffer;
static util::OutputBuffer error_buffer;

std::ostream log(&log_buffer);
std::ostream replay(&replay_buffer);
std::ostream error(&error_buffer);

namespace util {

std::string IOManager::s_root_io_dir = "./";

std::string IOManager::s_log_filename;
std::string IOManager::s_replay_filename;
std::string IOManager::s_error_filename;

std::ofstream* IOManager::s_log_ofstream;
std::ofstream* IOManager::s_replay_ofstream;

bool IOManager::s_initialized = false;

void
IOManager::setOutputDir(const std::string& dir)
{
  if (s_initialized) {
    UMPIRE_ERROR("Can't change IOManager output directory once initialized!");
  }

  s_root_io_dir = dir;
}

void
IOManager::initialize()
{
  if (!s_initialized) {
    s_log_filename = makeUniqueFilename(s_root_io_dir, "umpire", "log");
    s_replay_filename = makeUniqueFilename(s_root_io_dir, "umpire", "replay");
    s_error_filename = "";

    log_buffer.setConsoleStream(&std::cout);
    replay_buffer.setConsoleStream(nullptr);
    error_buffer.setConsoleStream(&std::cout);

    std::cout << s_log_filename << std::endl;

    if (!opendir(s_root_io_dir.c_str()))
    {
      mkdir(s_root_io_dir.c_str(), 0700);
    }

    s_log_ofstream = new std::ofstream(s_log_filename);

    if (*s_log_ofstream) {
      log_buffer.setFileStream(s_log_ofstream);
    } else {
      std::cerr << "EEROREUAEOUA" << std::endl;
    }

    s_replay_ofstream = new std::ofstream(s_replay_filename);

    if (*s_replay_ofstream) {
      replay_buffer.setFileStream(s_replay_ofstream);
    } else {
      std::cerr << "EEROREUAEOUA" << std::endl;
    }

    s_initialized = true;
  }
}

std::string
IOManager::makeUniqueFilename(
    const std::string& base_dir,
    const std::string& name, 
    const std::string& extension)
{
  int unique_id = -1;
  std::stringstream ss;
  std::string filename;

  do {
    ss.str("");
    ss.clear();
    unique_id++;
    ss << base_dir << "/" << name << "." << unique_id << "." << extension;
    filename = ss.str();
  } while (fileExists(filename));

  return filename;
}

inline bool 
IOManager::fileExists(const std::string& path)
{
  std::ifstream ifile(path.c_str());
  return ifile.good();
}

} // end of namespace util
} // end of namespace umpire
