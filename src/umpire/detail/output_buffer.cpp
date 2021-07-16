//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/detail/output_buffer.hpp"

#include <ostream>

namespace umpire {
namespace detail {

void
output_buffer::setConsoleStream(std::ostream* stream)
{
  if (stream) {
    d_console_stream = stream->rdbuf();
  } else {
    d_console_stream = nullptr;
  }
}

void
output_buffer::setFileStream(std::ostream* stream)
{
  if (stream) {
    d_file_stream = stream->rdbuf();
  } else {
    d_file_stream = nullptr;
  }
}

int
output_buffer::overflow(int ch)
{
  if (ch == EOF)
  {
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

int
output_buffer::sync()
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

output_buffer::~output_buffer()
{
  if (d_console_stream) {
    d_console_stream->pubsync();
  }

  if (d_file_stream) {
    d_file_stream->pubsync();
  }
}

} // end of namespace detail
} // end of namespace umpire
