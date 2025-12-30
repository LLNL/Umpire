//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/resource/MemoryResourceRegistry.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/CLI11/CLI11.hpp"

namespace {
const std::string hwm_name{"Memory Usage HSM"};

std::string pretty(std::size_t n)
{
  const char* scale[] = {
    " bytes",
    " kilobytes",
    " megabytes",
    " gigabytes",
    " terabytes"
  };

  int suffix{0};
  double count{static_cast<double>(n)};

  while (count > 1024) {
    count /= 1024;
    suffix++;
  }

  std::stringstream ss;

  ss << std::setprecision(2) << std::fixed << count << scale[suffix];

  return ss.str();
}

std::size_t get_memory_stat(const std::string& name)
{
  std::ifstream status{"/proc/self/status"};

  std::size_t rval{0};
  std::string line;
  while (std::getline(status, line)) {
    std::stringstream ss{line};
    std::string key;
    ss >> key;

    if (key == name) {
      std::size_t stat{0};
      ss >> stat;
      rval = std::size_t{stat * 1024};
      break;
    }
  }
  return rval;
}

};

struct MemoryStatSnapshot {
  void take_snapshot()
  {
    stat[hwm_name] = umpire::get_process_memory_usage_hwm();
    stat["VmPeak"] = get_memory_stat("VmPeak:");
    stat["VmSize"] = get_memory_stat("VmSize:");
    stat["VmHWM"] = get_memory_stat("VmHWM:");
    stat["VmRSS"] = get_memory_stat("VmRSS:");
    stat["VmData"] = get_memory_stat("VmData:");
    stat["VmStk"] = get_memory_stat("VmStk:");
    stat["VmExe"] = get_memory_stat("VmExe:");
    stat["VmLib"] = get_memory_stat("VmLib:");
    stat["VmPTE"] = get_memory_stat("VmPTE:");
  }

  std::unordered_map<std::string, std::size_t> stat{};
};

class AllocatorCostBenchmark {
public:
  friend std::ostream& operator<<(std::ostream& os, const AllocatorCostBenchmark& acb);

  AllocatorCostBenchmark(const std::string& a, std::size_t n, bool no_intro, bool _show_process_mem_info) :
    allocator_to_use{a}, number_of_allocs{n}, no_introspection{no_intro}, show_process_mem_info{_show_process_mem_info}
  {
    is_umpire_allocator = (allocator_to_use != "malloc");

    if (is_umpire_allocator) {
      umpire::resource::MemoryResourceRegistry& registry{umpire::resource::MemoryResourceRegistry::getInstance()};
      auto traits = registry.getDefaultTraitsForResource("HOST");
      traits.tracking = !no_introspection;
      umpire::Allocator resource_allocator = umpire::ResourceManager::getInstance().makeResource("HOST", traits);

      if (allocator_to_use == "Host") {
        umpire_allocator = resource_allocator;
      }
      else if (allocator_to_use == "Quick") {
        if (no_introspection) {
          umpire_allocator = umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::QuickPool, false>
            ("QUICK_POOL", resource_allocator);
        }
        else {
          umpire_allocator = umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::QuickPool, true>
            ("QUICK_POOL", resource_allocator);
        }
      }
    }
  }

  void run_space_benchmark()
  {
    std::vector<void*> slots;

    for (std::size_t i = 0; i < number_of_allocs; ++i)
      slots.push_back(nullptr);

    mstat_baseline.take_snapshot();

    auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < number_of_allocs; ++i)
      slots[i] = allocate(allocation_size);

    auto finish = std::chrono::high_resolution_clock::now();
    elapsed_time = finish - start;

    mstat.take_snapshot();

    for (std::size_t i = 0; i < number_of_allocs; ++i)
      deallocate(slots[i]);

  }

  void run_time_benchmark()
  {
    for (std::size_t i = 0; i < number_of_allocs; ++i)
      deallocate(allocate(allocation_size));
  }

  struct AllocatorValidator : public CLI::Validator {
    AllocatorValidator()
    {
      func_ = [](const std::string &str) {
        if (str != "Quick" && str != "Host" && str != "malloc") {
          return std::string("Invalid Allocator name, must be Host, Quick, or malloc");
        } else
          return std::string();
      };
    }
  };

private:
  const std::string allocator_to_use;
  const std::size_t number_of_allocs;
  const bool no_introspection;
  const bool show_process_mem_info;

  umpire::Allocator umpire_allocator;
  bool is_umpire_allocator;

  MemoryStatSnapshot mstat_baseline;
  MemoryStatSnapshot mstat;
  std::chrono::duration<double, std::milli> elapsed_time;
  const std::size_t allocation_size{4};

  void* allocate(std::size_t n)
  {
    void* ptr;

    if (is_umpire_allocator) {
      ptr = umpire_allocator.allocate(n);
    }
    else {
      ptr = ::malloc(n);
    }

    return ptr;
  }

  void deallocate(void* ptr)
  {
    if (is_umpire_allocator) {
      umpire_allocator.deallocate(ptr);
    }
    else {
      ::free(ptr);
    }
  }

};

std::ostream& operator<<(std::ostream& os, const AllocatorCostBenchmark& acb)
{
  os << acb.number_of_allocs << " 4-byte allocs from " << acb.allocator_to_use;
  if (acb.is_umpire_allocator)
    os << (acb.no_introspection ? std::string{"{Intro OFF}"} : std::string{"{Intro ON}"});

  if (acb.show_process_mem_info) {
    os << std::endl;
    for ( auto& m : acb.mstat.stat ) {
      const std::size_t total{m.second};
      std::unordered_map<std::string,std::size_t>::const_iterator base_it = acb.mstat_baseline.stat.find(m.first);
      const std::size_t baseline{base_it->second};
      const std::size_t cost{total - baseline};

      os
        << m.first << " "
        << "Total{" << pretty(total) << "}, "
        << "Baseline{" << pretty(baseline) << "}, "
        << "Cost{" << pretty(cost) << "}, "
        << "or {" << pretty(cost / acb.number_of_allocs) << "}/allocation"
        << std::endl;
    }
  }
  else {
    std::unordered_map<std::string,std::size_t>::const_iterator total_it = acb.mstat.stat.find(hwm_name);
    std::unordered_map<std::string,std::size_t>::const_iterator base_it = acb.mstat_baseline.stat.find(hwm_name);
    const std::size_t total{total_it->second};
    const std::size_t baseline{base_it->second};
    const std::size_t cost{total - baseline};
    double allocs = static_cast<double>(acb.number_of_allocs);
    os << " uses " << pretty(cost) << " of memory, costing "
      << pretty(cost / acb.number_of_allocs) << ", and "
      << std::chrono::duration_cast<std::chrono::nanoseconds>(acb.elapsed_time).count() / allocs << " nanoseconds per allocation"
      << std::endl;
  }

  return os;
}

int main(int argc, char* argv[])
{
  const static AllocatorCostBenchmark::AllocatorValidator valid_allocator;

  CLI::App app{"Benchmark the memory overhead per allocation of an Allocator"};

  std::string allocator;
  app.add_option("-a,--use-allocator", allocator,
      "Specify Allocator to use: 'Host', 'Quick', or 'malloc'.\n"
      "When 'malloc' is specified, use the raw system call (no Umpire).\n"
      "When 'Host' is specified, use HOST Umpire resource allocator.\n"
      "When 'Quick' is specified, use the Umpire Quickpool allocator.\n"
      )
    ->required()
    ->check(valid_allocator);

  std::size_t allocations;
  app.add_option("-n,--num-allocations", allocations, "Specify number of allocations to perform")
    ->required()
    ->check(CLI::Range(0, 100000000));

  bool no_introspection{false};
  app.add_flag("--no_introspection", no_introspection, "Disable introspection");

  bool show_process_mem_info{false};
  app.add_flag("--show_process_mem_info", show_process_mem_info, "Display process memory details");

  bool measure_time_overhead{false};
  app.add_flag("--measure_time_overhead", measure_time_overhead, "Measure time overhead");

  bool measure_space_overhead{false};
  app.add_flag("--measure_space_overhead", measure_space_overhead, "Measure space overhead");

  CLI11_PARSE(app, argc, argv);

  AllocatorCostBenchmark bm{allocator, allocations, no_introspection, show_process_mem_info};

  if (measure_time_overhead) {
     bm.run_time_benchmark();
  }
  else if (measure_space_overhead) {
     bm.run_space_benchmark();
     std::cout << bm;
  }

  return 0;
}
