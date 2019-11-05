//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "ReplayFile.hpp"
#include "ReplayMacros.hpp"

#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

ReplayFile::ReplayFile( std::string in_file_name )
  : m_in_file_name{in_file_name}, m_bin_file_name{m_in_file_name + m_bin_suffix}
{
  struct stat sbuf;

  if (stat(m_in_file_name.c_str(), &sbuf))
      REPLAY_ERROR( "Unable to open " << m_in_file_name );

  //
  // Assume our binary file will be <= size of the text file
  //
  max_file_size = sizeof(ReplayFile::Header) + sbuf.st_size;

  m_fd = open(  m_bin_file_name.c_str()
              , O_CREAT | O_RDWR
              , static_cast<mode_t>(0660));

  if (m_fd < 0)
    REPLAY_ERROR( "Unable to create: " << m_bin_file_name );

  if (fstat(m_fd, &sbuf))
    REPLAY_ERROR( "Unable to determine size of: " << m_bin_file_name);

  int flags;
  const int prot = PROT_READ|PROT_WRITE;

  if (sbuf.st_size < max_file_size) {
    m_compile_needed = true;
    flags = MAP_SHARED;   // Writes will make it to backing store

    if (lseek(m_fd, max_file_size-1, SEEK_SET) < 0)
      REPLAY_ERROR("lseek failed on " << m_bin_file_name);

    if (write(m_fd, "", 1) < 0)
      REPLAY_ERROR("write failed to " << m_bin_file_name);
  }
  else {
    m_compile_needed = false;
    flags = MAP_PRIVATE;  // Writes won't make it to backing store
  }

  m_op_tables = static_cast<ReplayFile::Header*>(mmap(nullptr, max_file_size, prot, flags, m_fd, 0));

  if (m_op_tables == MAP_FAILED)
    REPLAY_ERROR( "Unable to mmap to: " << m_bin_file_name );
}

ReplayFile::~ReplayFile()
{
  if (m_op_tables != nullptr && m_op_tables != MAP_FAILED ) {
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

