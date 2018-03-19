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
#ifndef UMPIRE_CnmemAllocator_HPP
#define UMPIRE_CnmemAllocator_HPP

#include "umpire/tpl/cnmem/cnmem.h"

#include <cstring>

namespace umpire {
namespace alloc {

struct CnmemAllocator
{
  const char* cnmemGetErrorString(cnmemStatus_t error) {
    switch (error) {
      case CNMEM_STATUS_SUCCESS: return "SUCCESS";
      case CNMEM_STATUS_NOT_INITIALIZED: return "NOT INITIALIZED";
      case CNMEM_STATUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
      case CNMEM_STATUS_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
      case CNMEM_STATUS_CUDA_ERROR: return cudaGetErrorString(cudaPeekAtLastError());
      case CNMEM_STATUS_UNKNOWN_ERROR: return "UNKNOWN";
      default: return "UNKNOWN";
    }

    return "UNKNOWN";
  }

  void* allocate(size_t bytes)
  {
    static bool initialized = false;

    if (!initialized) {
      cudaDeviceProp props;
      cudaGetDeviceProperties(&props, 0);
      cnmemDevice_t cnmem_device;
      std::memset(&cnmem_device, 0, sizeof(cnmem_device));
      cnmem_device.size = static_cast<size_t>(0.8 * props.totalGlobalMem);
      cnmemInit(1, &cnmem_device, CNMEM_FLAGS_DEFAULT);

      initialized = true;
    }

    void* ptr;
    cnmemStatus_t error = ::cnmemMalloc(&ptr, bytes, NULL);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != CNMEM_STATUS_SUCCESS) {
      UMPIRE_ERROR("cnmemMalloc( bytes = " << bytes << " ) failed with error: " << cnmemGetErrorString(error));
    } else {
      return ptr;
    }
  }

  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    cnmemStatus_t error = ::cnmemFree(ptr, NULL);
    if ( error != CNMEM_STATUS_SUCCESS ) {
      UMPIRE_ERROR("cnmemFree( ptr = " << ptr << " ) failed with error: " << cnmemGetErrorString(error));
    }
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CnmemAllocator_HPP
