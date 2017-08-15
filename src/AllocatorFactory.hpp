#ifndef UMPIRE_AllocatorFactory_HPP
#define UMPIRE_AllocatorFactory_HPP

#include "umpire/Allocator.hpp"

#include <memory>
#include <string>

namespace umpire {

class AllocatorFactory {
  public:
    virtual bool isValidAllocatorFor(const std::string& name) = 0;
    virtual std::shared_ptr<Allocator> create() = 0;
};

} // end of namespace umpire

#endif // UMPIRE_AllocatorFactory_HPP
