//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <sstream>

#include "gtest/gtest.h"
#include "umpire/util/OutputBuffer.hpp"

class OutputBufferTest : public ::testing::Test {
 protected:
  OutputBufferTest()
  {
    d_buffer.setConsoleStream(&d_mock_console_stream);
    d_buffer.setFileStream(&d_mock_file_stream);
  }

  std::stringstream d_mock_console_stream;
  std::stringstream d_mock_file_stream;

  umpire::util::OutputBuffer d_buffer;
};

TEST_F(OutputBufferTest, WriteToStreams)
{
  const std::string expected{"TEST"};

  d_buffer.overflow('T');
  d_buffer.overflow('E');
  d_buffer.overflow('S');
  d_buffer.overflow('T');

  d_buffer.sync();

  ASSERT_EQ(d_mock_console_stream.str(), expected);
  ASSERT_EQ(d_mock_file_stream.str(), expected);
}
