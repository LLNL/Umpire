//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_OutputBuffer_HPP
#define UMPIRE_OutputBuffer_HPP

#include <streambuf>

namespace umpire {
namespace util {

class OutputBuffer : public std::streambuf
{
public:
  OutputBuffer() = default;

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
