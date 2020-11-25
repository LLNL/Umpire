//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Zeroer_HPP
#define UMPIRE_Zeroer_HPP


namespace umpire {
namespace strategy {

class FixedPool;

namespace mixins {

class Zeroer
{
  public:
    Zeroer();

    void* allocateZero();
    bool deallocateZero(void* ptr);

  private:
    FixedPool* m_zero_byte_pool;

};

} // end of namespace mixins
} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_Zeroer_HPP
