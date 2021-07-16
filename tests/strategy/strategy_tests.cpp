//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/config.hpp"

#include "umpire/alloc.hpp"
#include "umpire/resource.hpp"
#include "umpire/strategy.hpp"


#include "gtest/gtest.h"

template <typename T>
class strategy_test : public ::testing::Test {
};

inline std::string get_unique_name() {
  static std::size_t id{0};

  return "strategy_" + id;
}

template<typename Strategy, typename Memory>
struct construct {};

template<typename Memory>
struct construct<umpire::strategy::fixed_pool<>, Memory> {
  static constexpr auto make(Memory* m) {
    return umpire::strategy::fixed_pool<>{get_unique_name(), m, 64};
  };
};

template<typename Memory>
struct construct<umpire::strategy::fixed_pool<Memory>, Memory> {
  static constexpr auto make(Memory* m) {
    return umpire::strategy::fixed_pool<Memory>{get_unique_name(), m, 64};
  };
};

template<typename Memory>
struct construct<umpire::strategy::slot_pool<>, Memory> {
  static constexpr auto make(Memory* m) {
    return umpire::strategy::slot_pool<>{get_unique_name(), m, 128};
  };
};

template<typename Memory>
struct construct<umpire::strategy::slot_pool<Memory>, Memory> {
  static constexpr auto make(Memory* m) {
    return umpire::strategy::slot_pool<Memory>{get_unique_name(), m, 128};
  };
};

template<typename Memory>
struct construct<umpire::strategy::quick_pool<>, Memory> {
  static constexpr auto make(Memory* m) {
    return umpire::strategy::quick_pool<>{get_unique_name(), m};
  };
};

template<typename Memory>
struct construct<umpire::strategy::quick_pool<Memory>, Memory> {
  static constexpr auto make(Memory* m) {
    return umpire::strategy::quick_pool<Memory>{get_unique_name(), m};
  };
};

template<typename Memory>
struct construct<umpire::strategy::dynamic_pool_map<>, Memory> {
  static constexpr auto make(Memory* m) {
    return umpire::strategy::dynamic_pool_map<>{get_unique_name(), m};
  };
};

template<typename Memory>
struct construct<umpire::strategy::dynamic_pool_map<Memory>, Memory> {
  static constexpr auto make(Memory* m) {
    return umpire::strategy::dynamic_pool_map<Memory>{get_unique_name(), m};
  };
};

template<typename Memory>
struct construct<umpire::strategy::dynamic_pool_list<>, Memory> {
  static constexpr auto make(Memory* m) {
    return umpire::strategy::dynamic_pool_list<>{get_unique_name(), m};
  };
};

template<typename Memory>
struct construct<umpire::strategy::dynamic_pool_list<Memory>, Memory> {
  static constexpr auto make(Memory* m) {
    return umpire::strategy::dynamic_pool_list<Memory>{get_unique_name(), m};
  };
};

template<typename Memory>
struct construct<umpire::strategy::mixed_pool<>, Memory> {
  static constexpr auto make(Memory* m) {
    return umpire::strategy::mixed_pool<>{get_unique_name(), m};
  };
};

template<typename Memory>
struct construct<umpire::strategy::mixed_pool<Memory>, Memory> {
  static constexpr auto make(Memory* m) {
    return umpire::strategy::mixed_pool<Memory>{get_unique_name(), m};
  };
};

TYPED_TEST_SUITE_P(strategy_test);

TYPED_TEST_P(strategy_test, constructors)
{
  using Strategy = typename std::tuple_element<0, TypeParam>::type;
  using Memory = typename std::tuple_element<1, TypeParam>::type;

  // Strategy strategy{Memory::get(), 16};
  Strategy strategy = construct<Strategy, Memory>::make(Memory::get());

  EXPECT_EQ(strategy.get_platform(), Memory::get()->get_platform());

  if (!std::is_same<typename Strategy::platform, umpire::resource::undefined_platform>::value) {
    EXPECT_TRUE(
      (std::is_same<typename Strategy::platform, typename Memory::platform>::value) 
    );
  } else {
    EXPECT_FALSE(
      (std::is_same<typename Strategy::platform, typename Memory::platform>::value) 
    );
  }
  EXPECT_EQ(strategy.get_platform(), Memory::get()->get_platform());
}

TYPED_TEST_P(strategy_test, allocate_deallocate)
{
  using Strategy = typename std::tuple_element<0, TypeParam>::type;
  using Memory = typename std::tuple_element<1, TypeParam>::type;

  Strategy strategy = construct<Strategy, Memory>::make(Memory::get());

  {
    auto ptr = strategy.allocate(64);

    EXPECT_NE(nullptr, ptr);

    strategy.deallocate(ptr);
  }

  SUCCEED();
}

REGISTER_TYPED_TEST_SUITE_P(
    strategy_test,
    constructors,
    allocate_deallocate);

using test_types = ::testing::Types<
        std::tuple<umpire::strategy::slot_pool<>, umpire::resource::host_memory<>>
      , std::tuple<umpire::strategy::slot_pool<umpire::resource::host_memory<>>, umpire::resource::host_memory<>>
      , std::tuple<umpire::strategy::fixed_pool<>, umpire::resource::host_memory<>>
      , std::tuple<umpire::strategy::fixed_pool<umpire::resource::host_memory<>>, umpire::resource::host_memory<>>
      , std::tuple<umpire::strategy::quick_pool<>, umpire::resource::host_memory<>>
      , std::tuple<umpire::strategy::quick_pool<umpire::resource::host_memory<>>, umpire::resource::host_memory<>>
      , std::tuple<umpire::strategy::dynamic_pool_map<>, umpire::resource::host_memory<>>
      , std::tuple<umpire::strategy::dynamic_pool_map<umpire::resource::host_memory<>>, umpire::resource::host_memory<>>
      , std::tuple<umpire::strategy::dynamic_pool_list<>, umpire::resource::host_memory<>>
      , std::tuple<umpire::strategy::dynamic_pool_list<umpire::resource::host_memory<>>, umpire::resource::host_memory<>>
      , std::tuple<umpire::strategy::mixed_pool<>, umpire::resource::host_memory<>>
      , std::tuple<umpire::strategy::mixed_pool<umpire::resource::host_memory<>>, umpire::resource::host_memory<>>
#if defined(UMPIRE_ENABLE_CUDA)
#endif
#if defined(UMPIRE_ENABLE_HIP)
#endif
>;

INSTANTIATE_TYPED_TEST_SUITE_P(_, strategy_test, test_types,);
