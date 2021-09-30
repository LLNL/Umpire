//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/event/recorder_factory.hpp"

#include "umpire/config.hpp"
#include "umpire/event/json_file_store.hpp"
#ifndef WIN32
#include "umpire/event/quest_database.hpp"
#endif
#ifdef UMPIRE_ENABLE_SQLITE
#include "umpire/event/sqlite_database.hpp"
#endif // UMPIRE_ENABLE_SQLITE
#include "umpire/util/io.hpp"

#if !defined(_MSC_VER)
#include <unistd.h> // getpid()
#else
#include <process.h>
#define getpid _getpid
#include <direct.h>
#endif

namespace umpire {
namespace event {

store_type& recorder_factory::get_recorder()
{
  static const std::string filename{
      util::make_unique_filename(util::get_io_output_dir(), util::get_io_output_basename(), getpid(), "stats")};
  static json_file_store db{filename};

  // static quest_database db{"localhost", "9009", "db"};
  // static binary_file_database db{"test.bin"};
#ifdef UMPIRE_ENABLE_SQLITE
  // static sqlite_database db{"test.db"};
#endif // UMPIRE_ENABLE_SQLITE
  static event_store_recorder recorder(&db);

  return recorder;
}

} // namespace event
} // namespace umpire
