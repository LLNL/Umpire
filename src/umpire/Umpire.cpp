//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/Umpire.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"
#include "camp/resource/platform.hpp"
#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
  #include <cuda_runtime_api.h>
#endif

//make sure "/opt/rocm-3.6.0/hsa/include/" is 
//added to blt/cmake/thirdparty/SetupHIP.cmake file
#if defined(UMPIRE_ENABLE_HIP)
  #include <hip/hip_runtime.h>
#endif

#if !defined(_MSC_VER)
#include <unistd.h>
#endif
#include <fstream>
#include <sstream>

UMPIRE_EXPORT volatile int umpire_ver_4_found = 0;

using myResource = umpire::MemoryResourceTraits::resource_type;
using cPlatform = camp::resources::Platform;

namespace umpire {

void print_allocator_records(Allocator allocator, std::ostream& os)
{
  std::stringstream ss;
  auto& rm = umpire::ResourceManager::getInstance();

  auto strategy = allocator.getAllocationStrategy();

  rm.m_allocations.print(
      [strategy](const util::AllocationRecord& rec) {
        return rec.strategy == strategy;
      },
      ss);

  if (!ss.str().empty()) {
    os << "Allocations for " << allocator.getName()
       << " allocator:" << std::endl
       << ss.str() << std::endl;
  }
}

std::vector<util::AllocationRecord> get_allocator_records(Allocator allocator)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto strategy = allocator.getAllocationStrategy();

  std::vector<util::AllocationRecord> recs;
  std::copy_if(rm.m_allocations.begin(), rm.m_allocations.end(),
               std::back_inserter(recs),
               [strategy](const util::AllocationRecord& rec) {
                 return rec.strategy == strategy;
               });

  return recs;
}

bool pointer_overlaps(void* left_ptr, void* right_ptr)
{
  auto& rm = umpire::ResourceManager::getInstance();

  try {
    auto left_record = rm.findAllocationRecord(left_ptr);
    auto right_record = rm.findAllocationRecord(right_ptr);

    char* left{reinterpret_cast<char*>(left_record->ptr)};
    char* right{reinterpret_cast<char*>(right_record->ptr)};

    return ((right >= left) && ((left + left_record->size) > right) &&
            ((right + right_record->size) > (left + left_record->size)));
  } catch (umpire::util::Exception&) {
    UMPIRE_LOG(Error, "Unknown pointer in pointer_overlaps");
    throw;
  }
}

bool pointer_contains(void* left_ptr, void* right_ptr)
{
  auto& rm = umpire::ResourceManager::getInstance();

  try {
    auto left_record = rm.findAllocationRecord(left_ptr);
    auto right_record = rm.findAllocationRecord(right_ptr);

    char* left{reinterpret_cast<char*>(left_record->ptr)};
    char* right{reinterpret_cast<char*>(right_record->ptr)};

    return ((right >= left) && (left + left_record->size > right) &&
            (right + right_record->size <= left + left_record->size));
  } catch (umpire::util::Exception&) {
    UMPIRE_LOG(Error, "Unknown pointer in pointer_contains");
    throw;
  }
}

bool is_host_pageable()
{
#if defined(UMPIRE_ENABLE_CUDA)
  int pageableMem = 0;
  int cdev = 0;
  cudaGetDevice(&cdev);

  //Device supports coherently accessing pageable memory
  //without calling cudaHostRegister on it
  cudaDeviceGetAttribute(&pageableMem,
    cudaDevAttrPageableMemoryAccess, cdev); 

  if(pageableMem)
    return true;
  else
    return false;
#endif
#if defined(UMPIRE_ENABLE_HIP)
  hipDeviceProp_t props;
  int hdev = 0;
  hipGetDevice(&hdev);

  //Check whether HIP can map host memory.
  hipGetDeviceProperties(&props, hdev);
  if(props.canMapHostMemory)
    return true;
  else
    return false;
#endif
  return false;
}

bool is_accessible(Platform p, Allocator a) 
{
 /*
  * This function implements the following table which 
  * describes whether or not an Umpire::Allocator (rows,
  * labeled by memory resource_type) is accessible by a CAMP
  * platform (columns). 'T' = True; 'F' = False; 'X' = Does
  * Not Exist; '*' = Conditional. For those marked with '*',
  * certain tests are done to determine accessibility. See
  * confluence page for more details.
  * 
  *               UNDEFINED  HOST  CUDA  OMP_TARGET  HIP  SYCL   
  *  UNKNOWN          F       F     F        F        F    F
  *  HOST             F       T     *        T        *    F
  *  DEVICE           F       *     T        T        T    T
  *  DEVICE_CONST     F       F     T        X        T    X 
  *  UM               F       T     T        X        T    T 
  *  PINNED           F       T     T        X        T    T
  *  FILE             F       T     F        F        F    F
  */

  switch(p) {
    case (cPlatform::host):
    {
      if(get_resource(a) == myResource::UNKNOWN
       || get_resource(a) == myResource::DEVICE_CONST)
        return false;

#if defined(UMPIRE_ENABLE_CUDA)
      //check DEVICE case for (Umpire) CUDA
      return is_host_pageable();
#elif defined(UMPIRE_ENABLE_HIP)
      //check DEVICE case for (Umpire) HIP
      return is_host_pageable();
#endif

      //If it reaches here, no DEVICE resource is accessible
      if (get_resource(a) == myResource::DEVICE)
        return false;
      else
        return true;       
    }
    break;
    ////////////////////////////////////////////////////
    case (cPlatform::cuda): 
    {
#if defined(UMPIRE_ENABLE_CUDA)    
      if (get_resource(a) == myResource::UNKNOWN 
       || get_resource(a) == myResource::FILE)
        return false;
      else if(get_resource(a) == myResource::HOST)
        return is_host_pageable();
      else
        return true;
#else
      std::cout << "Platform is undefined" << std::endl;
      return false;
#endif
    }
    break;
    ////////////////////////////////////////////////////
    case (cPlatform::omp_target):
    {
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
      if (get_resource(a) == myResource::UNKNOWN
       || get_resource(a) == myResource::FILE)
        return false;
      else
        return true; //true for (Umpire) HOST or DEVICE
#else
      std::cout << "Platform is undefined" << std::endl;
      return false;
#endif
    }
    break;
    ////////////////////////////////////////////////////
    case (cPlatform::hip):
    {
#if defined(UMPIRE_ENABLE_HIP)
      if (get_resource(a) == myResource::UNKNOWN 
       || get_resource(a) == myResource::FILE)
        return false;
      else if(get_resource(a) == myResource::HOST)      
        return is_host_pageable();
      else
        return true;
#else
      std::cout << "Platform is undefined" << std::endl;
      return false;
#endif
    }
    break;
    ////////////////////////////////////////////////////
    case (cPlatform::sycl):
    {
#if defined(UMPIRE_ENABLE_SYCL)
      if (get_resource(a) == myResource::DEVICE
       || get_resource(a) == myResource::UM
       || get_resource(a) == myResource::PINNED)
        return true;
      else
        return false;
#else
      std::cout << "Platform is undefined" << std::endl;
      return false;
#endif
    }
    break;
    ////////////////////////////////////////////////////
    default:
    {
      std::cout << "Platform is undefined" << std::endl;
      return false;
    }
    break;
  }
  return false;
}

MemoryResourceTraits::resource_type get_resource(Allocator a) 
{
  return a.getAllocationStrategy()->getTraits().resource;
}

std::string get_backtrace(void* ptr)
{
#if defined(UMPIRE_ENABLE_BACKTRACE)
  auto& rm = umpire::ResourceManager::getInstance();
  auto record = rm.findAllocationRecord(ptr);
  return umpire::util::backtracer<>::print(record->allocation_backtrace);
#else
  UMPIRE_USE_VAR(ptr);
  return "[Umpire: UMPIRE_BACKTRACE=Off]";
#endif
}

std::size_t get_process_memory_usage()
{
#if defined(_MSC_VER) || defined(__APPLE__)
  return 0;
#else
  std::size_t ignore;
  std::size_t resident;
  std::ifstream statm("/proc/self/statm");
  statm >> ignore >> resident >> ignore;
  statm.close();
  long page_size{::sysconf(_SC_PAGE_SIZE)};
  return std::size_t{resident * page_size};
#endif
}

std::size_t get_device_memory_usage(int device_id)
{
#if defined(UMPIRE_ENABLE_CUDA)
  std::size_t mem_free{0};
  std::size_t mem_tot{0};

  int current_device;
  cudaGetDevice(&current_device);

  cudaSetDevice(device_id);

  cudaMemGetInfo(&mem_free, &mem_tot);

  cudaSetDevice(current_device);

  return std::size_t{mem_tot - mem_free};
#else
  UMPIRE_USE_VAR(device_id);
  return 0;
#endif
}

std::vector<util::AllocationRecord> get_leaked_allocations(Allocator allocator)
{
  return get_allocator_records(allocator);
}

} // end namespace umpire
