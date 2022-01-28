#ifndef HeterogeneousCore_CUDAUtilities_SharedEventPtr_h
#define HeterogeneousCore_CUDAUtilities_SharedEventPtr_h

#include <memory>
#include <type_traits>

#include <hip/hip_runtime.h>

namespace cms {
  namespace hip {
    // cudaEvent_t itself is a typedef for a pointer, for the use with
    // edm::ReusableObjectHolder the pointed-to type is more interesting
    // to avoid extra layer of indirection
    using SharedEventPtr = std::shared_ptr<std::remove_pointer_t<hipEvent_t>>;
  }  // namespace cuda
}  // namespace cms

#endif
