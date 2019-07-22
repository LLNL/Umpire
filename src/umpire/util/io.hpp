//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_IOManager_HPP
#define UMPIRE_IOManager_HPP

#include <string>
#include <ostream>

namespace umpire {

// Output streams
std::ostream& log();
std::ostream& replay();
std::ostream& error();

namespace util {

/*!
 * \brief Initialize the streams. This method is called when ResourceManger is
 * initialized. Do not call this manually.
 */
void initialize_io(const bool enable_log, const bool enable_replay);

} // end namespace util
} // end namespace umpire

#endif // UMPIRE_IOManager_HPP
