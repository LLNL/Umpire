//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_OutputBuffer_HPP
#define UMPIRE_OutputBuffer_HPP

#include <streambuf>

namespace umpire {
namespace util {

class OutputBuffer : public std::streambuf {
 public:
  OutputBuffer() = default;

  ~OutputBuffer();

  void setConsoleStream(std::ostream* stream);
  void setFileStream(std::ostream* stream);

  int overflow(int ch) override;
  int sync() override;

 private:
  std::streambuf* d_console_stream;
  std::streambuf* d_file_stream;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_OutputBuffer_HPP
