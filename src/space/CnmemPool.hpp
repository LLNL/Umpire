#ifndef UMPIRE_CnmemPool_HPP_
#define UMPIRE_CnmemPool_HPP_

#include "umpire/space/MemorySpace.hpp"

namespace umpire {
namespace space {

class CnmemPool 
  //: public MemorySpace
{
  public: 
    CnmemPool(const std::string& name);

    virtual void* allocate(size_t bytes);
    virtual void free(void* ptr);
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_CnmemPool_HPP_
