#ifndef HeterogenousCore_HIPUtilities_deviceCount_h
#define HeterogenousCore_HIPUtilities_deviceCount_h

#include "HIPCore/cudaCheck.h"

#include <hip/hip_runtime.h>

namespace cms {
  namespace hip {
    inline int deviceCount() {
      int ndevices;
      cudaCheck(hipGetDeviceCount(&ndevices));
      return ndevices;
    }
  }  // namespace hip
}  // namespace cms

#endif
