//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#ifndef UMPIRE_OperationRegistry_HPP
#define UMPIRE_OperationRegistry_HPP

#include "umpire/op/MemoryOperation.hpp"

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Platform.hpp"

#include <memory>
#include <unordered_map>
#include <functional>

namespace umpire {
namespace op {

struct pair_hash {
  std::size_t operator () (const std::pair<Platform, Platform> &p) const noexcept {
      auto h1 = std::hash<int>{}(static_cast<int>(p.first));
      auto h2 = std::hash<int>{}(static_cast<int>(p.second));

      // Mainly for demonstration purposes, i.e. works but is overly simple
      // In the real world, use sth. like boost.hash_combine
      return h1 ^ h2;
  }
};

/*!
 * \brief The MemoryOperationRegistry serves as a registry for MemoryOperation
 * objects. It is a singleton class, typically accessed through the
 * ResourceManager.
 *
 * The MemoryOperationRegistry class provides lookup mechanisms allowing
 * searching for the appropriate MemoryOperation to be applied to allocations
 * made with particular AllocationStrategy objects.
 *
 * MemoryOperations provided by Umpire are registered with the
 * MemoryOperationRegistry when it is constructed. Additional MemoryOperations
 * can be registered later using the registerOperation method.
 *
 * The following operations are pre-registered for all AllocationStrategy pairs:
 * - "COPY"
 * - "MEMSET"
 * - "REALLOCATE"
 *
 * \see MemoryOperation
 * \see AllocationStrategy
 */
class MemoryOperationRegistry {
  public:
    /*!
     * \brief Get the MemoryOperationRegistry singleton instance.
     */
    static MemoryOperationRegistry& getInstance() noexcept;

    /*!
     * \brief Function to find a MemoryOperation object
     *
     * Finds the MemoryOperation object that matches the given name and
     * AllocationStrategy objects. If the requested MemoryOperation is not
     * found, this method will throw an Exception.
     *
     * \param name Name of operation.
     * \param src_allocator AllocationStrategy of the source allocation.
     * \param dst_allocator AllocationStrategy of the destination allocation.
     *
     * \throws umpire::util::Exception if the requested MemoryOperation is not
     *         found.
     */
    std::shared_ptr<umpire::op::MemoryOperation> find(
        const std::string& name,
        const std::shared_ptr<strategy::AllocationStrategy>& source_allocator,
        const std::shared_ptr<strategy::AllocationStrategy>& dst_allocator);

    /*!
     * \brief Add a new MemoryOperation to the registry
     *
     * This object will register the provided MemoryOperation, making it
     * available for later retrieval using MemoryOperation::find
     *
     * \param name Name of the operation.
     * \param platforms pair of Platforms for the source and destination.
     * \param operation pointer to the MemoryOperation.
     */
    void registerOperation(
      const std::string& name,
      std::pair<Platform, Platform> platforms,
      std::shared_ptr<MemoryOperation>&& operation) noexcept;

  protected:
    MemoryOperationRegistry() noexcept;
    MemoryOperationRegistry (const MemoryOperationRegistry&) = delete;
    MemoryOperationRegistry& operator= (const MemoryOperationRegistry&) = delete;

  private:
    static MemoryOperationRegistry* s_memory_operation_registry_instance;

    /*
     * Doubly-nested unordered_map that stores MemoryOperations by first name,
     * then by Platform pair.
     */
    std::unordered_map<
      std::string,
      std::unordered_map< std::pair<Platform, Platform>,
                          std::shared_ptr<MemoryOperation>,
                          pair_hash > > m_operators;

};

} // end of namespace op
} // end of namespace umpire

#endif
