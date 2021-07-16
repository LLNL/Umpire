#pragma once

#include "umpire/config.hpp"

#include "umpire/op/operations.hpp"
#include "umpire/op/host.hpp"
#if defined(UMPIRE_ENABLE_CUDA)
#include "umpire/op/cuda.hpp"
#endif
#if defined(UMPIRE_ENABLE_HIP)
#include "umpire/op/hip.hpp"
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
#include "umpire/op/omp_target.hpp"
#endif

#include "umpire/op/dispatch.hpp"