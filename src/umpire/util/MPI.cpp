#include "umpire/config.hpp"

#include "umpire/util/MPI.hpp" 

#include "umpire/util/Macros.hpp"

#if defined(UMPIRE_ENABLE_MPI)
#include "mpi.h"
#endif

namespace umpire {
namespace util {

int MPI::s_rank = -1;
int MPI::s_world_size = -1;
bool MPI::s_initialized = false;

void 
MPI::initialize()
{
  if (!s_initialized) {
#if !defined(UMPIRE_ENABLE_MPI)
    s_rank = 0;
    s_world_size = 1;
#else
    MPI_Comm_rank(MPI_COMM_WORLD, &s_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &s_world_size);
#endif

    s_initialized = true;
  } else {
    UMPIRE_ERROR("umpire::MPI already initialized, cannot call initialize() again!");
  }
}

void
MPI::finalize()
{
  if (s_initialized) {
#if !defined(UMPIRE_ENABLE_MPI)
  s_rank = -1;
  s_world_size = -1;
#endif
  } else {
    UMPIRE_ERROR("Cannot call MPI::finalize() when umpire::MPI not initialiazed");
  }
}

int
MPI::getRank()
{
  return s_rank;
}

int
MPI::getSize()
{
  return s_world_size;
}

} // end of namespace util
} // end of namespace umpire
