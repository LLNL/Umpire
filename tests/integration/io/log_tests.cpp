//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

int main(int UMPIRE_UNUSED_ARG(argc), char** UMPIRE_UNUSED_ARG(argv))
{
  auto& rm = umpire::ResourceManager::getInstance();
  UMPIRE_USE_VAR(rm);

  return 0;
}
