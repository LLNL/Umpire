//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_IOManager_HPP
#define UMPIRE_IOManager_HPP

#include <string>
#include <ostream>

namespace umpire {
namespace util {

class IOManager {
public:
  static void initialize(bool enable_log, bool enable_replay);

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
