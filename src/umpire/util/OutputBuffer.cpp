//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/OutputBuffer.hpp"

#include <ostream>

namespace umpire {
namespace util {

void OutputBuffer::setConsoleStream(std::ostream* stream)
{
  if (stream) {
    d_console_stream = stream->rdbuf();
  } else {
    d_console_stream = nullptr;
  }
}

void OutputBuffer::setFileStream(std::ostream* stream)
{
  if (stream) {
    d_file_stream = stream->rdbuf();
  } else {
    d_file_stream = nullptr;
  }
}

int OutputBuffer::overflow(int ch)
{
  if (ch == EOF) {
    return !EOF;
  } else {
    int r_console{ch};
    int r_file{ch};

    if (d_console_stream) {
      r_console = d_console_stream->sputc(static_cast<char>(ch));
    }

    if (d_file_stream) {
      r_file = d_file_stream->sputc(static_cast<char>(ch));
    }

    return r_console == EOF || r_file == EOF ? EOF : ch;
  }
}

int OutputBuffer::sync()
{
  auto ret = 0;

  if (d_console_stream) {
    ret = d_console_stream->pubsync();
  }

  if (d_file_stream) {
    ret += d_file_stream->pubsync();
  }

  return ret == 0 ? 0 : -1;
}

OutputBuffer::~OutputBuffer()
{
  if (d_console_stream) {
    d_console_stream->pubsync();
  }

  if (d_file_stream) {
    d_file_stream->pubsync();
  }
}

} // end of namespace util
} // end of namespace umpire
