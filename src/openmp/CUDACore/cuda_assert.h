// The omission of #include guards is on purpose: it does make sense to #include
// this file multiple times, setting a different value of GPU_DEBUG beforehand.

#ifdef __CUDA_ARCH__
// hack to address "Undefined reference to '__assert_fail'" linker error
#include <stdio.h>
#undef assert
#define assert(x) \
  if (!(x))       \
    printf("assert failed\n");

#else
#include <cassert>
#endif  // __CUDA_ARCH__
