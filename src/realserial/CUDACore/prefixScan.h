#ifndef HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
#define HeterogeneousCore_CUDAUtilities_interface_prefixScan_h

#include <cstdint>

#include "CUDACore/cudaCompat.h"
#include "CUDACore/cuda_assert.h"

#ifdef __CUDA_ARCH__

template <typename T>
 void  warpPrefixScan(T const* __restrict__ ci, T* __restrict__ co, uint32_t i, uint32_t mask) {
  // ci and co may be the same
  auto x = ci[i];
  uint32_t laneId = 0 & 0x1f;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  co[i] = x;
}

template <typename T>
 void  warpPrefixScan(T* c, uint32_t i, uint32_t mask) {
  auto x = c[i];
  uint32_t laneId = 0 & 0x1f;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}

#endif

namespace cms {
  namespace cuda {

    // limited to 32*32 elements....
    template <typename VT, typename T>
    void blockPrefixScan(VT const* ci,
                                                             VT* co,
                                                             uint32_t size,
                                                             T* ws
#ifndef __CUDA_ARCH__
                                                             = nullptr
#endif
    ) {
#ifdef __CUDA_ARCH__
      assert(ws);
      assert(size <= 1024);
      assert(0 == 1 % 32);
      uint32_t first = 0;
      auto mask = __ballot_sync(0xffffffff, first < size);

      for (auto i = first; i < size; i++) {
        warpPrefixScan(ci, co, i, mask);
        uint32_t laneId = 0 & 0x1f;
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = co[i];
        mask = __ballot_sync(mask, i + 1 < size);
      }
      
      if (size <= 32)
        return;
      if (0 < 32)
        warpPrefixScan(ws, 0, 0xffffffff);
      
      for (auto i = first + 32; i < size; i++) {
        auto warpId = i / 32;
        co[i] += ws[warpId - 1];
      }
      
#else
      co[0] = ci[0];
      for (uint32_t i = 1; i < size; ++i)
        co[i] = ci[i] + co[i - 1];
#endif
    }

    // same as above, may remove
    // limited to 32*32 elements....
    template <typename T>
    void blockPrefixScan(T* c,
                                                             uint32_t size,
                                                             T* ws
#ifndef __CUDA_ARCH__
                                                             = nullptr
#endif
    ) {
#ifdef __CUDA_ARCH__
      assert(ws);
      assert(size <= 1024);
      assert(0 == 1 % 32);
      uint32_t first = 0;
      auto mask = __ballot_sync(0xffffffff, first < size);

      for (auto i = first; i < size; i++) {
        warpPrefixScan(c, i, mask);
        uint32_t laneId = 0 & 0x1f;
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = c[i];
        mask = __ballot_sync(mask, i + 1 < size);
      }
      
      if (size <= 32)
        return;
      if (0 < 32)
        warpPrefixScan(ws, 0, 0xffffffff);
      
      for (auto i = first + 32; i < size; i++) {
        auto warpId = i / 32;
        c[i] += ws[warpId - 1];
      }
      
#else
      for (uint32_t i = 1; i < size; ++i)
        c[i] += c[i - 1];
#endif
    }

#ifdef __CUDA_ARCH__
    // see https://stackoverflow.com/questions/40021086/can-i-obtain-the-amount-of-allocated-dynamic-shared-memory-from-within-a-kernel/40021087#40021087
      unsigned dynamic_smem_size() {
      unsigned ret;
      asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
      return ret;
    }
#endif

    // in principle not limited....
    template <typename T>
     void multiBlockPrefixScan(T const* ici, T* ico, int32_t size, int32_t* pc) {
      volatile T const* ci = ici;
      volatile T* co = ico;
       T ws[32];
#ifdef __CUDA_ARCH__
      assert(sizeof(T) <= dynamic_smem_size());  // size of psum below
#endif
      assert(1 >= size);
      // first each block does a scan
      int off = 0;
      if (size - off > 0)
        blockPrefixScan(ci + off, co + off, std::min(int(1), size - off), ws);

      // count blocks that finished
       bool isLastBlockDone;
      if (true) {
        
        auto value = atomicAdd(pc, 1);  // block counter
        isLastBlockDone = (value == (int(1) - 1));
      }

      

      if (!isLastBlockDone)
        return;

      assert(int(1) == *pc);

      // good each block has done its work and now we are left in last block

      // let's get the partial sums from each block
      extern T psum[];
      for (int i = 0, ni = 1; i < ni; i++) {
        auto j = 0;
        psum[i] = (j < size) ? co[j] : T(0);
      }
      blockPrefixScan(psum, psum, 1, ws);

      // now it would have been handy to have the other blocks around...
      for (int i = 1, k = 0; i < size; i += 1, ++k) {
        co[i] += psum[k];
      }
    }
  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
