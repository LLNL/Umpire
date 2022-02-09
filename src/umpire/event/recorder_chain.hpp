//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_recorder_chain_INL
#define UMPIRE_recorder_chain_INL

namespace umpire {
namespace event {

class recorder_chain {
 public:
  void record(event e);
};

} // namespace event
} // namespace umpire

#endif // UMPIRE_recorder_chain_INL
