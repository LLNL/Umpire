//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "ReplayFile.hpp"
#include "ReplayMacros.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

ReplayFile::ReplayFile( std::string in_file_name )
  : m_in_file_name{in_file_name}, m_bin_file_name{m_in_file_name + m_bin_suffix}
{
  struct stat statbuf;
  int flags;

  if (stat(m_bin_file_name.c_str(), &statbuf)) {
    m_fd = open(m_bin_file_name.c_str(), O_CREAT | O_RDWR, S_IRWXU | S_IRWXG);

    if (m_fd < 0)
      REPLAY_ERROR( "Unable to create: " << m_bin_file_name );

    flags = MAP_SHARED;
  }
  else {
    m_fd = open(m_bin_file_name.c_str(), O_RDWR);

    if (m_fd < 0)
      REPLAY_ERROR( "Unable to open: " << m_bin_file_name );

    flags = MAP_PRIVATE;
  }

  m_op_tables = static_cast<ReplayFile::Header*>(
      mmap(nullptr, max_file_size, PROT_READ|PROT_WRITE, flags, m_fd, 0));

  if (m_op_tables == MAP_FAILED)
    REPLAY_ERROR( "Unable to mmap to: " << m_bin_file_name );
}

ReplayFile::~ReplayFile()
{
  if (m_op_tables != nullptr) {
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
  dest[source.length()] = '\n';
}

