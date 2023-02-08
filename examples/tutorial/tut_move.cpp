//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

double* move_data(double* source_data, const std::string& destination)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto dest_allocator = rm.getAllocator(destination);

  std::cout << "Moved source data (" << source_data << ") to destination ";

  // _sphinx_tag_tut_move_start
  double* dest_data = static_cast<double*>(rm.move(source_data, dest_allocator));
  // _sphinx_tag_tut_move_end

  std::cout << destination << " (" << dest_data << ")" << std::endl;

  return dest_data;
}

int main(int, char**)
{
  constexpr std::size_t SIZE = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");

  double* data = static_cast<double*>(allocator.allocate(SIZE * sizeof(double)));

  std::cout << "Allocated " << (SIZE * sizeof(double)) << " bytes using the " << allocator.getName() << " allocator."
            << std::endl;

  std::cout << "Filling with 0.0...";

  for (std::size_t i = 0; i < SIZE; i++) {
    data[i] = 0.0;
  }

  std::cout << "done." << std::endl;

  data = move_data(data, "HOST");
#if defined(UMPIRE_ENABLE_DEVICE)
  data = move_data(data, "DEVICE");
#endif
#if defined(UMPIRE_ENABLE_UM)
  data = move_data(data, "UM");
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  data = move_data(data, "PINNED");
#endif

  rm.deallocate(data);

  return 0;
}
