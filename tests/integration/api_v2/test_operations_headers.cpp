//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/op/copy.hpp"
#include "umpire/op/memset.hpp"
#include "umpire/op/prefetch.hpp"
#include "umpire/op/reallocate.hpp"

#include <gtest/gtest.h>

namespace {

void compile_operation_headers(void* src, void* dst, void** ptr, camp::resources::Resource& resource)
{
  umpire::copy(src, dst, 0);
  (void)umpire::copy(src, dst, 0, resource);

  umpire::memset(src, 0, 0);
  (void)umpire::memset(src, 0, 0, resource);

  (void)umpire::reallocate(ptr, 0);
  (void)umpire::reallocate(ptr, 0, resource);

  umpire::prefetch(src, 0, 0);
  (void)umpire::prefetch(src, 0, 0, resource);
}

TEST(ApiV2OperationHeaders, StandaloneHeadersCompile)
{
  auto* usage = &compile_operation_headers;
  (void)usage;
  SUCCEED();
}

} // namespace
