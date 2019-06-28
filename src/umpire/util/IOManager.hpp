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
#ifndef UMPIRE_IOManager_HPP
#define UMPIRE_IOManager_HPP

#include <string>
#include <ostream>

namespace umpire {
namespace util {

class IOManager {
public:
  static void initialize();

  static void finalize();

  static void setOutputDir(const std::string& dir);
private:
  static std::string s_root_io_dir;
  static std::string s_file_basename;

  static std::string s_log_filename;
  static std::string s_replay_filename;
  static std::string s_error_filename;

  static std::ofstream* s_log_ofstream;
  static std::ofstream* s_replay_ofstream;

  static bool s_initialized;

};

} /* namespace util */

extern std::ostream log;
extern std::ostream replay;
extern std::ostream error;


} /* namespace umpire */

#endif /* UMPIRE_IOManager_HPP */
