#ifndef UMPIRE_MemoryResourceFactory_HPP
#define UMPIRE_MemoryResourceFactory_HPP

#include "umpire/resource/MemoryResource.hpp"

#include <memory>
#include <string>

namespace umpire {
namespace resource {

class MemoryResourceFactory {
  public:
    virtual bool isValidMemoryResourceFor(const std::string& name) = 0;
    virtual std::shared_ptr<MemoryResource> create(const std::string& name, int id) = 0;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_MemoryResourceFactory_HPP
