//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "camp/camp.hpp"
#include "gtest/gtest.h"
#include "umpire/config.hpp"

template <typename T>
struct tag_to_string {
};

struct host_resource_tag {
};

template <>
struct tag_to_string<host_resource_tag> {
  static constexpr const char* value = "HOST";
};

#if defined(UMPIRE_ENABLE_DEVICE)
struct device_resource_tag {
};

template <>
struct tag_to_string<device_resource_tag> {
  static constexpr const char* value = "DEVICE";
};
#endif

#if defined(UMPIRE_ENABLE_UM)
struct um_resource_tag {
};

template <>
struct tag_to_string<um_resource_tag> {
  static constexpr const char* value = "UM";
};
#endif

#if defined(UMPIRE_ENABLE_PINNED)
struct pinned_resource_tag {
};

template <>
struct tag_to_string<pinned_resource_tag> {
  static constexpr const char* value = "PINNED";
};
#endif

#if defined(UMPIRE_ENABLE_FILE_RESOURCE)
struct file_resource_tag {
};

template <>
struct tag_to_string<file_resource_tag> {
  static constexpr const char* value = "FILE";
};
#endif

#if defined(UMPIRE_ENABLE_CONST)
struct device_const_resource_tag {
};

template <>
struct tag_to_string<device_const_resource_tag> {
  static constexpr const char* value = "DEVICE_CONST";
};
#endif

template <class T>
struct Test;

template <class... T>
struct Test<camp::list<T...>> {
  using Types = ::testing::Types<T...>;
};
