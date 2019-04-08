//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2019, Lawrence Livermore National Security, LLC.
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
#ifndef UMPIRE_Tools_InputParser_HPP
#define UMPIRE_Tools_InputParser_HPP

#include <algorithm>
#include <string>
#include <vector>

namespace umpire {

namespace tools {

// From: https://stackoverflow.com/questions/865668/how-to-parse-command-line-arguments-in-c

class InputParser {
public:
  InputParser (int &argc, char **argv) {
    for (int i=1; i < argc; ++i)
      this->tokens.push_back(std::string(argv[i]));
  }

  const std::string& get_command_option(const std::string &option) const {
    std::vector<std::string>::const_iterator itr;
    itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
    if (itr != this->tokens.end() && ++itr != this->tokens.end()){
      return *itr;
    }
    static const std::string empty_string("");
    return empty_string;
  }

  bool command_option_exists(const std::string &option) const {
    return std::find(this->tokens.begin(), this->tokens.end(), option)
                    != this->tokens.end();
  }

private:
  std::vector <std::string> tokens;
};







} // end of namespace tools

} // end of namespace umpire

#endif // UMPIRE_Tools_InputParser_HPP
