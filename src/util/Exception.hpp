#ifndef UMPIRE_Exception_HPP
#define UMPIRE_Exception_HPP

#include <string>
#include <exception>

namespace umpire {
namespace util {

class Exception : public std::exception {
  public:
    Exception(const std::string& msg,
        const std::string &file,
        int line);

    std::string message() const;
    virtual const char* what() const throw();

  private:
    std::string m_message;
    std::string m_file;
    int m_line;

    std::string m_what;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_Exception_HPP
