//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
#include <limits>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include "ReplayFile.hpp"
#include "ReplayMacros.hpp"

ReplayFile::ReplayFile( bool force_compile, std::string input_filename, std::string binary_filename )
  : m_force_compile{force_compile}, m_input_filename{input_filename}, m_binary_filename{binary_filename}
{
  const int prot = PROT_READ|PROT_WRITE;
  int flags;

  m_fd = open(m_binary_filename.c_str(), O_CREAT | O_RDWR, static_cast<mode_t>(0660));

  if (m_fd < 0)
    REPLAY_ERROR( "Unable to create: " << m_binary_filename );

  checkHeader();

  if ( compileNeeded() ) {
    flags = MAP_SHARED;   // Writes will make it to backing store

    if (lseek(m_fd, max_file_size-1, SEEK_SET) < 0)
      REPLAY_ERROR("lseek failed on " << m_binary_filename);

    if (write(m_fd, "", 1) < 0)
      REPLAY_ERROR("write failed to " << m_binary_filename);
  }
  else {
    flags = MAP_PRIVATE;
  }

  m_op_tables = static_cast<ReplayFile::Header*>(mmap(nullptr, max_file_size, prot, flags, m_fd, 0));
  if (m_op_tables == MAP_FAILED)
    REPLAY_ERROR( "Unable to mmap to: " << m_binary_filename );

  m_op_tables->m.magic = REPLAY_MAGIC;
  m_op_tables->m.version = REPLAY_VERSION;
}

std::string ReplayFile::getLine(std::size_t lineno)
{
  std::ifstream file{m_input_filename};

  if ( ! file.is_open() ) {
    REPLAY_ERROR("Unable to open input file " << m_input_filename);
  }

  file.seekg(std::ios::beg);
  for (std::size_t i=0; i < lineno - 1; ++i) {
      file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
  }

  std::string line;
  std::getline(file, line);
  return line;
}

ReplayFile::~ReplayFile()
{
  if (m_op_tables != nullptr && m_op_tables != MAP_FAILED ) {
    if ( compileNeeded() ) {
      off_t actual_size = sizeof(Header) + (m_op_tables->num_operations * sizeof(Operation));

      if (ftruncate(m_fd, actual_size) < 0)
        REPLAY_ERROR( "Failed to truncate file size for " << m_binary_filename);
    }

    munmap(m_op_tables, max_file_size);
    m_op_tables = nullptr;
  }

  close(m_fd);
}

ReplayFile::Header* ReplayFile::getOperationsTable()
{
  return m_op_tables;
}

void ReplayFile::copyString(std::string source, char (&dest)[max_name_length])
{
  strncpy( dest, source.c_str(), source.length() );
  dest[source.length()] = '\0';
}

void ReplayFile::checkHeader()
{
  struct stat sbuf;
  Header::Magic m;

  if (! m_force_compile ) {
    if (read(m_fd, &m, sizeof(m)) == sizeof(m)) {
      if (m.magic == REPLAY_MAGIC) {
        if (m.version == REPLAY_VERSION) {
          m_compile_needed = false;

          if (stat(m_binary_filename.c_str(), &sbuf))
            REPLAY_ERROR( "Unable to open " << m_binary_filename );

          max_file_size = sbuf.st_size;
          return;
        }
      }
    }
  }

  m_compile_needed = true;

  if (stat(m_input_filename.c_str(), &sbuf))
    REPLAY_ERROR( "Unable to open " << m_input_filename );

  max_file_size = sizeof(ReplayFile::Header) + sbuf.st_size;

}
#endif
