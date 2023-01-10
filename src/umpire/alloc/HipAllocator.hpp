//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipAllocator_HPP
#define UMPIRE_HipAllocator_HPP

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"
#include "umpire/strategy/GranularityController.hpp"

namespace umpire {
namespace alloc {

struct HipAllocator {
  enum Granularity {
    fine_grain_coherence = 1,
    course_grain_coherence = 2
  };

  void set_granularity(umpire::strategy::GranularityController::Granularity g) {
    switch(g) {
        case umpire::strategy::GranularityController::Granularity::FineGrainedCoherence:
        m_granularity = fine_grain_coherence;
        break;
        case umpire::strategy::GranularityController::Granularity::CourseGrainedCohorence:
        m_granularity = course_grain_coherence;
        break;
      default:
        UMPIRE_ERROR(runtime_error, umpire::fmt::format("Unknown coherence granularity: {}", g));
        break;
    }
  }

  Granularity m_granularity{fine_grain_coherence};
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_HipAllocator_HPP
