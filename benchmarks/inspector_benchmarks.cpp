//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <memory>

#include "benchmark/benchmark.h"

#include "umpire/config.hpp"

#include "umpire/util/Macros.hpp"

#include "umpire/strategy/mixins/Inspector.hpp"

static const int MAX_REGISTRATIONS = 100000;

void fill_ptr(uintptr_t ptr[])
{
  for (int i{0}; i < MAX_REGISTRATIONS; ++i) ptr[i] = (i+1) * sizeof(int);
}

void register_all(umpire::strategy::mixins::Inspector& inspector, uintptr_t ptr[])
{
  for (int i{0}; i < MAX_REGISTRATIONS; ++i)
    inspector.registerAllocation(reinterpret_cast<void*>(ptr[i]), sizeof(int), nullptr);
}

void deregister_all(umpire::strategy::mixins::Inspector& inspector, uintptr_t ptr[])
{
  for (int i{0}; i < MAX_REGISTRATIONS; ++i)
    inspector.deregisterAllocation(reinterpret_cast<void*>(ptr[i]));
}

class InspectorRegister : public ::benchmark::Fixture {
public:
  using ::benchmark::Fixture::SetUp;
  // using ::benchmark::Fixture::TearDown;

  InspectorRegister() { fill_ptr(ptr); }

  void TearDown(const ::benchmark::State& UMPIRE_UNUSED_ARG(st)) {
    // deregister_all(inspector, ptr);
  }

  umpire::strategy::mixins::Inspector inspector;
  uintptr_t ptr[MAX_REGISTRATIONS];
};


BENCHMARK_F(InspectorRegister, add)(benchmark::State &st)
{
  int i{0};
  while(st.KeepRunning()) {
    if (i == MAX_REGISTRATIONS) {
      st.PauseTiming();
      deregister_all(inspector, ptr);
      i = 0;
      st.ResumeTiming();
    }
    inspector.registerAllocation(reinterpret_cast<void*>(ptr[i++]), sizeof(int), nullptr);
  }
}


class InspectorDeregister : public ::benchmark::Fixture {
public:
  // using ::benchmark::Fixture::SetUp;
  // using ::benchmark::Fixture::TearDown;

  InspectorDeregister() { fill_ptr(ptr); }

  void SetUp(const ::benchmark::State& UMPIRE_UNUSED_ARG(st)) {
    register_all(inspector, ptr);
  }

  void TearDown(const ::benchmark::State& UMPIRE_UNUSED_ARG(st)) {
    // deregister_all(inspector, ptr);
  }

  umpire::strategy::mixins::Inspector inspector;
  uintptr_t ptr[MAX_REGISTRATIONS];
};


BENCHMARK_F(InspectorDeregister, remove)(benchmark::State &st)
{
  int i{0};
  while(st.KeepRunning()) {
    if (i == MAX_REGISTRATIONS) {
      st.PauseTiming();
      register_all(inspector, ptr);
      i = 0;
      st.ResumeTiming();
    }
    inspector.deregisterAllocation(reinterpret_cast<void*>(ptr[i++]));
  }
}


// // BENCHMARK_DEFINE_F(InspectorBenchmark, add   )(benchmark::State &st) { add(st);    }
// // BENCHMARK_DEFINE_F(InspectorBenchmark, remove)(benchmark::State &st) { remove(st); }

// BENCHMARK_REGISTER_F(InspectorBenchmark, add);
// BENCHMARK_REGISTER_F(InspectorBenchmark, remove);

BENCHMARK_MAIN();
