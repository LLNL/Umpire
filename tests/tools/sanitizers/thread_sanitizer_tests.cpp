//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/host_memory.hpp"
#include "umpire/strategy/monotonic_buffer.hpp"

#include <iostream>
#include <string>
#include <thread>
#include <vector>

void run_monotonic_buffer_race()
{
  auto& host = umpire::resource::host_memory<>::get();

  constexpr int num_threads = 8;
  constexpr int iterations = 2048;
  constexpr std::size_t allocation_size = 64;
  constexpr std::size_t capacity = num_threads * iterations * allocation_size * 2;

  umpire::strategy::monotonic_buffer<umpire::resource::host_memory<>> buffer(
    "thread_sanitizer_monotonic_buffer", &host, capacity);

  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back([&buffer, t]() {
      for (int i = 0; i < iterations; ++i) {
        auto* ptr = static_cast<char*>(buffer.allocate(allocation_size));
        ptr[0] = static_cast<char>(t + i);
      }
    });
  }

  for (auto& thread : workers) {
    thread.join();
  }
}

int main(int argc, char* argv[])
{
  const std::string mode = argc > 1 ? argv[1] : "monotonic_buffer";

  if (mode == "monotonic_buffer") {
    run_monotonic_buffer_race();
    return 0;
  }

  std::cerr << "Unknown TSan test mode: " << mode << std::endl;
  return 2;
}
