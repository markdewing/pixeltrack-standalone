#ifndef HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
#define HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "CUDACore/AtomicPairCounter.h"
#include "CUDACore/cuda_assert.h"
#include "CUDACore/cudastdAlgorithm.h"
#include "CUDACore/prefixScan.h"

namespace cms {
  namespace openmp {

    template <typename Histo, typename T>
     void countFromVector(Histo *__restrict__ h,
                                    uint32_t nh,
                                    T const *__restrict__ v,
                                    uint32_t const *__restrict__ offsets) {
      int first = 0;
      for (int i = first, nt = offsets[nh]; i < nt; i++) {
        auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
        assert((*off) > 0);
        int32_t ih = off - offsets - 1;
        assert(ih >= 0);
        assert(ih < int(nh));
        (*h).count(v[i], ih);
      }
    }

    template <typename Histo, typename T>
     void fillFromVector(Histo *__restrict__ h,
                                   uint32_t nh,
                                   T const *__restrict__ v,
                                   uint32_t const *__restrict__ offsets) {
      int first = 0;
      for (int i = first, nt = offsets[nh]; i < nt; i++) {
        auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
        assert((*off) > 0);
        int32_t ih = off - offsets - 1;
        assert(ih >= 0);
        assert(ih < int(nh));
        (*h).fill(v[i], i, ih);
      }
    }

    template <typename Histo>
    inline  void launchZero(Histo *__restrict__ h) {
      uint32_t *poff = (uint32_t *)((char *)(h) + offsetof(Histo, off));
      int32_t size = offsetof(Histo, bins) - offsetof(Histo, off);
      assert(size >= int(sizeof(uint32_t) * Histo::totbins()));
      ::memset(poff, 0, size);
    }

    template <typename Histo>
    inline  void launchFinalize(Histo *__restrict__ h) {
      h->finalize();
    }

    template <typename Histo, typename T>
    inline  void fillManyFromVector(Histo *__restrict__ h,
                                                                  uint32_t nh,
                                                                  T const *__restrict__ v,
                                                                  uint32_t const *__restrict__ offsets,
                                                                  uint32_t totSize,
                                                                  int nthreads) {
      launchZero(h);
      countFromVector(h, nh, v, offsets);
      h->finalize();
      fillFromVector(h, nh, v, offsets);
    }

    template <typename Assoc>
     void finalizeBulk(AtomicPairCounter const *apc, Assoc *__restrict__ assoc) {
      assoc->bulkFinalizeFill(*apc);
    }

    // iteratate over N bins left and right of the one containing "v"
    template <typename Hist, typename V, typename Func>
       void forEachInBins(Hist const &hist, V value, int n, Func func) {
      int bs = Hist::bin(value);
      int be = std::min(int(Hist::nbins() - 1), bs + n);
      bs = std::max(0, bs - n);
      assert(be >= bs);
      for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
        func(*pj);
      }
    }

    // iteratate over bins containing all values in window wmin, wmax
    template <typename Hist, typename V, typename Func>
       void forEachInWindow(Hist const &hist, V wmin, V wmax, Func const &func) {
      auto bs = Hist::bin(wmin);
      auto be = Hist::bin(wmax);
      assert(be >= bs);
      for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
        func(*pj);
      }
    }

    template <typename T,                  // the type of the discretized input values
              uint32_t NBINS,              // number of bins
              uint32_t SIZE,               // max number of element
              uint32_t S = sizeof(T) * 8,  // number of significant bits in T
              typename I = uint32_t,  // type stored in the container (usually an index in a vector of the input values)
              uint32_t NHISTS = 1     // number of histos stored
              >
    class HistoContainer {
    public:
      using Counter = uint32_t;

      using CountersOnly = HistoContainer<T, NBINS, 0, S, I, NHISTS>;

      using index_type = I;
      using UT = typename std::make_unsigned<T>::type;

      static constexpr uint32_t ilog2(uint32_t v) {
        constexpr uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
        constexpr uint32_t s[] = {1, 2, 4, 8, 16};

        uint32_t r = 0;  // result of log2(v) will go here
        for (auto i = 4; i >= 0; i--)
          if (v & b[i]) {
            v >>= s[i];
            r |= s[i];
          }
        return r;
      }

      static constexpr uint32_t sizeT() { return S; }
      static constexpr uint32_t nbins() { return NBINS; }
      static constexpr uint32_t nhists() { return NHISTS; }
      static constexpr uint32_t totbins() { return NHISTS * NBINS + 1; }
      static constexpr uint32_t nbits() { return ilog2(NBINS - 1) + 1; }
      static constexpr uint32_t capacity() { return SIZE; }

      static constexpr auto histOff(uint32_t nh) { return NBINS * nh; }

      static constexpr UT bin(T t) {
        constexpr uint32_t shift = sizeT() - nbits();
        constexpr uint32_t mask = (1 << nbits()) - 1;
        return (t >> shift) & mask;
      }

        void zero() {
        for (auto &i : off)
          i = 0;
      }

         void add(CountersOnly const &co) {
        for (uint32_t i = 0; i < totbins(); ++i) {
          auto &a = (std::atomic<Counter> &)(off[i]);
          a += co.off[i];
        }
      }

      static    uint32_t atomicIncrement(Counter &x) {
        auto &a = (std::atomic<Counter> &)(x);
        return a++;
      }

      static    uint32_t atomicDecrement(Counter &x) {
        auto &a = (std::atomic<Counter> &)(x);
        return a--;
      }

         void countDirect(T b) {
        assert(b < nbins());
        atomicIncrement(off[b]);
      }

         void fillDirect(T b, index_type j) {
        assert(b < nbins());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

         int32_t bulkFill(AtomicPairCounter &apc, index_type const *v, uint32_t n) {
        auto c = apc.add(n);
        if (c.m >= nbins())
          return -int32_t(c.m);
        off[c.m] = c.n;
        for (uint32_t j = 0; j < n; ++j)
          bins[c.n + j] = v[j];
        return c.m;
      }

         void bulkFinalize(AtomicPairCounter const &apc) {
        off[apc.get().m] = apc.get().n;
      }

         void bulkFinalizeFill(AtomicPairCounter const &apc) {
        auto m = apc.get().m;
        auto n = apc.get().n;
        if (m >= nbins()) {  // overflow!
          off[nbins()] = uint32_t(off[nbins() - 1]);
          return;
        }
        auto first = m + 0;
        for (auto i = first; i < totbins(); i++) {
          off[i] = n;
        }
      }

         void count(T t) {
        uint32_t b = bin(t);
        assert(b < nbins());
        atomicIncrement(off[b]);
      }

         void fill(T t, index_type j) {
        uint32_t b = bin(t);
        assert(b < nbins());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

         void count(T t, uint32_t nh) {
        uint32_t b = bin(t);
        assert(b < nbins());
        b += histOff(nh);
        assert(b < totbins());
        atomicIncrement(off[b]);
      }

         void fill(T t, index_type j, uint32_t nh) {
        uint32_t b = bin(t);
        assert(b < nbins());
        b += histOff(nh);
        assert(b < totbins());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

         void finalize(Counter *ws = nullptr) {
        assert(off[totbins() - 1] == 0);
        blockPrefixScan(off, totbins(), ws);
        assert(off[totbins() - 1] == off[totbins() - 2]);
      }

      constexpr auto size() const { return uint32_t(off[totbins() - 1]); }
      constexpr auto size(uint32_t b) const { return off[b + 1] - off[b]; }

      constexpr index_type const *begin() const { return bins; }
      constexpr index_type const *end() const { return begin() + size(); }

      constexpr index_type const *begin(uint32_t b) const { return bins + off[b]; }
      constexpr index_type const *end(uint32_t b) const { return bins + off[b + 1]; }

      Counter off[totbins()];
      int32_t psws;  // prefix-scan working space
      index_type bins[capacity()];
    };

    template <typename I,        // type stored in the container (usually an index in a vector of the input values)
              uint32_t MAXONES,  // max number of "ones"
              uint32_t MAXMANYS  // max number of "manys"
              >
    using OneToManyAssoc = HistoContainer<uint32_t, MAXONES, MAXMANYS, sizeof(uint32_t) * 8, I, 1>;

  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
