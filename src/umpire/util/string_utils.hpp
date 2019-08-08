//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_STRING_UTILS_HPP
#define UMPIRE_STRING_UTILS_HPP

#include <string>
#include <cctype>     // for std::toupper
#include <algorithm>  // for std::equal

namespace umpire {
namespace util {

inline int 
case_insensitive_match(const std::string s1, const std::string s2) {
    return (s1.size() == s2.size()) &&
        std::equal(s1.begin(), s1.end(), s2.begin(), [] (char c1, char c2) {
            return (std::toupper(c1) == std::toupper(c2));
    });
}

}
}
#endif // UMPIRE_STRING_UTILS_HPP