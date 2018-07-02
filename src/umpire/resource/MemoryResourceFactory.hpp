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
    /*
     * \brief Check whether the MemoryResource constructed by this factory is
     * valid for the given name
     *
     * \return true if the MemoryResource matches name.
     */
    virtual bool isValidMemoryResourceFor(const std::string& name) = 0;

    /*!
     * \brief Construct a MemoryResource with the given name and id.
     *
     * \param name Name of the MemoryResource.
     * \param id ID of the MemoryResource.
     */
    virtual std::shared_ptr<MemoryResource> create(const std::string& name, int id) = 0;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_MemoryResourceFactory_HPP
