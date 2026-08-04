//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/FileMemoryResourceFactory.hpp"

#include <memory>

#include "umpire/resource/FileMemoryResource.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"

// v2's file_memory only supports plain mmap(MAP_SHARED) mappings, with no
// UMAP-backed path. v1's FileMemoryResource additionally supports UMAP when
// UMPIRE_ENABLE_UMAP is set (see FileMemoryResource.cpp), so delegation is
// excluded in that configuration and falls back to native.
#if defined(UMPIRE_V1_DELEGATE_TO_V2) && !defined(UMPIRE_ENABLE_UMAP)
#include "umpire/resource/file_memory.hpp"
#include "umpire/resource/v2_backed_resource.hpp"
#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif
#endif

namespace umpire {
namespace resource {

bool FileMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if (name.find("FILE") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> FileMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> FileMemoryResourceFactory::create(const std::string& name, int id,
                                                                            MemoryResourceTraits traits)
{
#if defined(UMPIRE_V1_DELEGATE_TO_V2) && !defined(UMPIRE_ENABLE_UMAP)
  // Tracking=false: see the double-tracking discussion in
  // v2_backed_resource.hpp.
  auto v2_memory = std::make_unique<resource::file_memory<false>>(name + "_v2backed");

  // Replicates FileMemoryResource::isAccessibleFrom()/isPageable(): host is
  // always accessible; cuda is accessible only if the active device reports
  // coherent pageable-memory access; everything else is inaccessible.
  return util::make_unique<v2_backed_resource>(name, id, traits, Platform::undefined, std::move(v2_memory),
                                                [](Platform p) {
                                                  if (p == Platform::host) {
                                                    return true;
                                                  }
#if defined(UMPIRE_ENABLE_CUDA)
                                                  else if (p == Platform::cuda) {
                                                    int pageable_mem = 0;
                                                    int cdev = 0;
                                                    cudaError_t err = cudaGetDevice(&cdev);
                                                    if (err != cudaSuccess) {
                                                      return false;
                                                    }
                                                    err = cudaDeviceGetAttribute(
                                                        &pageable_mem, cudaDevAttrPageableMemoryAccess, cdev);
                                                    if (err != cudaSuccess) {
                                                      return false;
                                                    }
                                                    return pageable_mem != 0;
                                                  }
#endif
                                                  else {
                                                    return false;
                                                  }
                                                });
#else
  return util::make_unique<FileMemoryResource>(Platform::undefined, name, id, traits);
#endif
}

MemoryResourceTraits FileMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 0;

  traits.vendor = MemoryResourceTraits::vendor_type::unknown;
  traits.kind = MemoryResourceTraits::memory_type::unknown;
  traits.used_for = MemoryResourceTraits::optimized_for::any;
  traits.resource = MemoryResourceTraits::resource_type::file;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
