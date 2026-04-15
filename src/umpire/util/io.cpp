//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/io.hpp"

#include <cstdlib>
#include <fstream>
#include <string>

#if defined(UMPIRE_ENABLE_FILESYSTEM)
#include <filesystem>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#if !defined(_MSC_VER)
#include <unistd.h> // getpid()
#else
#include <process.h>
#define getpid _getpid
#endif

namespace umpire {
namespace util {

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
#if defined(_MSC_VER)
    return (info.st_mode & _S_IFDIR) != 0;
#else
    return S_ISDIR(info.st_mode);
#endif
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
