//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_io_HPP
#define UMPIRE_io_HPP

#include <ostream>
#include <string>

namespace umpire {

// Output streams
std::ostream& log();
std::ostream& error();

namespace util {

std::string make_unique_filename(const std::string& base_dir, const std::string& name, const int pid,
                                 const std::string& extension);

bool file_exists(const std::string& file);

bool directory_exists(const std::string& file);

const std::string& get_io_output_dir();
const std::string& get_io_output_basename();

/*!
 * \brief Initialize the streams. This method is called when ResourceManager is
 * initialized. Most users will not need to call this manually.
 * \warning This function will capture references to buffers of std::cerr and/or std::cout. If these are using custom
 * buffers with explicitly-manager lifetime should may need to call this and finalize_io() to control
 * explicitly initialization/finalization of Umpire I/O.
 */
void initialize_io(const bool enable_log);

/*!
 * \brief Counterpart of initialize_io that finalizes the streams and ensures that no live references to the buffers
 * of standard streams exist. Most users will not need to call this manually.
 */
void finalize_io();

/*!
 * \brief Synchronize all stream buffers to their respective output sequences.
 * This function is usually called by exception generating code like
 * UMPIRE_ERROR.
 */
void flush_files();

} // end namespace util
} // end namespace umpire

#endif // UMPIRE_IOManager_HPP
