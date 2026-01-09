//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/HostMpi3SharedMemoryResource.hpp"

#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/MPI.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace resource {

HostMpi3SharedMemoryResource::HostMpi3SharedMemoryResource(const std::string& name, int id, MemoryResourceTraits traits)
    : MemoryResource{name, id, traits}
{
  constexpr int IGNORE_KEY{0};
  MPI_Comm_split_type(util::MPI::getCommunicator(), MPI_COMM_TYPE_SHARED, IGNORE_KEY, MPI_INFO_NULL, &m_shared_comm);
  MPI_Comm_rank(m_shared_comm, &m_local_rank);

  // Free the comm at exit during cleanup in MPI_Finalize. We pass the m_shared_comm
  // by turning it into an int (as for Fortran) and then decoding that in the callback.
  int keyval = 0;
  MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, free_comm, &keyval, nullptr);
  MPI_Comm_set_attr(MPI_COMM_SELF, keyval, (void*)(intptr_t)MPI_Comm_c2f(m_shared_comm));
}

HostMpi3SharedMemoryResource::~HostMpi3SharedMemoryResource()
{
  // NOTE: m_shared_comm is freed at cleanup pre MPI_Finalize
}

void* HostMpi3SharedMemoryResource::allocate(std::size_t bytes)
{
  void* ptr;
  MPI_Win win;
  MPI_Aint local_size = (m_local_rank != 0) ? 0 : bytes;
  MPI_Aint size = bytes;
  int disp{sizeof(unsigned char)};

  MPI_Win_allocate_shared(local_size, disp, MPI_INFO_NULL, m_shared_comm, &ptr, &win);
  MPI_Win_shared_query(win, 0, &size, &disp, &ptr);
  m_shared_windows[ptr] = win;

  return ptr;
}

void HostMpi3SharedMemoryResource::deallocate(void* ptr, std::size_t)
{
  auto window = m_shared_windows.find(ptr);
  if (window != m_shared_windows.end()) {
    MPI_Win_free(&(window->second));
    m_shared_windows.erase(window);
  } else {
    UMPIRE_ERROR(umpire::unknown_pointer_error, "");
  }
}

bool HostMpi3SharedMemoryResource::isAccessibleFrom(Platform p) noexcept
{
  if (p == Platform::host)
    return true;
  else
    return false;
}

Platform HostMpi3SharedMemoryResource::getPlatform() noexcept
{
  return Platform::host;
}

int HostMpi3SharedMemoryResource::free_comm(MPI_Comm UMPIRE_UNUSED_ARG(comm), int UMPIRE_UNUSED_ARG(keyval),
                                            void* attribute_val, void* UMPIRE_UNUSED_ARG(extra_state))
{
  // Interpret attribute_val as a MPI_Fint comm number.
  const auto comm_number = (MPI_Fint)(intptr_t)(attribute_val);
  MPI_Comm comm_to_free = MPI_Comm_f2c(comm_number);
  MPI_Comm_free(&comm_to_free);
  return MPI_SUCCESS;
}

} // end of namespace resource
} // end of namespace umpire
