//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryResourceFactory_HPP
#define UMPIRE_MemoryResourceFactory_HPP

#include "umpire/resource/MemoryResource.hpp"

#include <memory>
#include <string>

namespace umpire {
namespace resource {

/*!
 * \brief Abstract factory class for constructing MemoryResource objects.
 *
 * Concrete implementations of this class are used by the
 * MemoryResourceRegistry to construct MemoryResource objects.
 *
 * \see MemoryResourceRegistry
 */
class MemoryResourceFactory {
  public:
    virtual ~MemoryResourceFactory() = default;

    /*
     * \brief Check whether the MemoryResource constructed by this factory is
     * valid for the given name
     *
     * \return true if the MemoryResource matches name.
     */
    virtual bool isValidMemoryResourceFor(const std::string& name) noexcept = 0;

    /*!
     * \brief Construct a MemoryResource with the given name and id.
     *
     * \param name Name of the MemoryResource.
     * \param id ID of the MemoryResource.
     */
    virtual std::unique_ptr<resource::MemoryResource> create(const std::string& name, int id) = 0;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_MemoryResourceFactory_HPP
