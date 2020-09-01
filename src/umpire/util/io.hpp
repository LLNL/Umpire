//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_IOManager_HPP
#define UMPIRE_IOManager_HPP

#include <ostream>
#include <string>

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

/*!
 * \brief Synchronize all stream buffers to their respective output sequences.
 * This function is usually called by exception generating code like
 * UMPIRE_ERROR.
 */
void flush_files();

} // end namespace util
} // end namespace umpire

#endif // UMPIRE_IOManager_HPP
