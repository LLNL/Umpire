//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/host_memory.hpp"
#include "umpire/strategy/thread_safe.hpp"

#include <functional>
#include <iostream>
#include <thread>

namespace {

void worker(umpire::strategy::thread_safe<umpire::resource::host_memory<>>& shared, int value)
{
  int* ptr = static_cast<int*>(shared.allocate(sizeof(int)));
  *ptr = value;
  std::cout << "worker value: " << *ptr << '\n';
  shared.deallocate(ptr);
}

} // namespace

int main()
{
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::thread_safe<umpire::resource::host_memory<>> shared{"SHARED_HOST", &host};

  std::thread first(worker, std::ref(shared), 7);
  std::thread second(worker, std::ref(shared), 11);

  first.join();
  second.join();

  return 0;
}
