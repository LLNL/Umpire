//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
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
    inspector.deregisterAllocation(reinterpret_cast<void*>(ptr[i]), nullptr);
}

class InspectorRegister : public ::benchmark::Fixture {
public:
  InspectorRegister() { fill_ptr(ptr); }

  using ::benchmark::Fixture::SetUp;
  using ::benchmark::Fixture::TearDown;

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
  InspectorDeregister() { fill_ptr(ptr); }

  using ::benchmark::Fixture::SetUp;
  using ::benchmark::Fixture::TearDown;

  void SetUp(const ::benchmark::State&) override final {
    register_all(inspector, ptr);
  }

  void TearDown(const ::benchmark::State&) override final {}

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
    inspector.deregisterAllocation(reinterpret_cast<void*>(ptr[i++]), nullptr);
  }
}

BENCHMARK_MAIN();
