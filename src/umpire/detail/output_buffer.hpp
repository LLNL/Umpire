//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include <streambuf>

namespace umpire {
namespace detail {

class output_buffer : public std::streambuf
{
public:
  output_buffer() = default;

  ~output_buffer();

  void setConsoleStream(std::ostream* stream);
  void setFileStream(std::ostream* stream);

  int overflow(int ch) override;
  int sync() override;

private:
  std::streambuf* d_console_stream;
  std::streambuf* d_file_stream;
};

} // end of namespace detail
} // end of namespace umpire
