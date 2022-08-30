#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>
#include <array>
#include <memory>

#include "CUDACore/HistoContainer.h"
using cms::openmp::AtomicPairCounter;

constexpr uint32_t MaxElem = 64000;
constexpr uint32_t MaxTk = 8000;
constexpr uint32_t MaxAssocs = 4 * MaxTk;

using Assoc = cms::openmp::OneToManyAssoc<uint16_t, MaxElem, MaxAssocs>;
using SmallAssoc = cms::openmp::OneToManyAssoc<uint16_t, 128, MaxAssocs>;
using Multiplicity = cms::openmp::OneToManyAssoc<uint16_t, 8, MaxTk>;
using TK = std::array<uint16_t, 4>;

 void countMultiLocal(TK const* __restrict__ tk, Multiplicity* __restrict__ assoc, int32_t n) {
  int first = 0;
  for (int i = first; i < n; i++) {
     Multiplicity::CountersOnly local;
    if (true)
      local.zero();
    
    local.countDirect(2 + i % 4);
    
    if (true)
      assoc->add(local);
  }
}

 void countMulti(TK const* __restrict__ tk, Multiplicity* __restrict__ assoc, int32_t n) {
  int first = 0;
  for (int i = first; i < n; i++)
    assoc->countDirect(2 + i % 4);
}

 void verifyMulti(Multiplicity* __restrict__ m1, Multiplicity* __restrict__ m2) {
  uint32_t first = 0;
  for (auto i = first; i < Multiplicity::totbins(); i++)
    assert(m1->off[i] == m2->off[i]);
}

 void count(TK const* __restrict__ tk, Assoc* __restrict__ assoc, int32_t n) {
  int first = 0;
  for (int i = first; i < 4 * n; i++) {
    auto k = i / 4;
    auto j = i - 4 * k;
    assert(j < 4);
    if (k >= n)
      return;
    if (tk[k][j] < MaxElem)
      assoc->countDirect(tk[k][j]);
  }
}

 void fill(TK const* __restrict__ tk, Assoc* __restrict__ assoc, int32_t n) {
  int first = 0;
  for (int i = first; i < 4 * n; i++) {
    auto k = i / 4;
    auto j = i - 4 * k;
    assert(j < 4);
    if (k >= n)
      return;
    if (tk[k][j] < MaxElem)
      assoc->fillDirect(tk[k][j], k);
  }
}

 void verify(Assoc* __restrict__ assoc) { assert(assoc->size() < Assoc::capacity()); }

template <typename Assoc>
 void fillBulk(AtomicPairCounter* apc, TK const* __restrict__ tk, Assoc* __restrict__ assoc, int32_t n) {
  int first = 0;
  for (int k = first; k < n; k++) {
    auto m = tk[k][3] < MaxElem ? 4 : 3;
    assoc->bulkFill(*apc, &tk[k][0], m);
  }
}

template <typename Assoc>
 void verifyBulk(Assoc const* __restrict__ assoc, AtomicPairCounter const* apc) {
  if (apc->get().m >= Assoc::nbins())
    printf("Overflow %d %d\n", apc->get().m, Assoc::nbins());
  assert(assoc->size() < Assoc::capacity());
}

int main() {
  std::cout << "OneToManyAssoc " << sizeof(Assoc) << ' ' << Assoc::nbins() << ' ' << Assoc::capacity() << std::endl;
  std::cout << "OneToManyAssoc (small) " << sizeof(SmallAssoc) << ' ' << SmallAssoc::nbins() << ' '
            << SmallAssoc::capacity() << std::endl;

  std::mt19937 eng;

  std::geometric_distribution<int> rdm(0.8);

  constexpr uint32_t N = 4000;

  std::vector<std::array<uint16_t, 4>> tr(N);

  // fill with "index" to element
  long long ave = 0;
  int imax = 0;
  auto n = 0U;
  auto z = 0U;
  auto nz = 0U;
  for (auto i = 0U; i < 4U; ++i) {
    auto j = 0U;
    while (j < N && n < MaxElem) {
      if (z == 11) {
        ++n;
        z = 0;
        ++nz;
        continue;
      }  // a bit of not assoc
      auto x = rdm(eng);
      auto k = std::min(j + x + 1, N);
      if (i == 3 && z == 3) {  // some triplets time to time
        for (; j < k; ++j)
          tr[j][i] = MaxElem + 1;
      } else {
        ave += x + 1;
        imax = std::max(imax, x);
        for (; j < k; ++j)
          tr[j][i] = n;
        ++n;
      }
      ++z;
    }
    assert(n <= MaxElem);
    assert(j <= N);
  }
  std::cout << "filled with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << nz << std::endl;

  Assoc la;
  SmallAssoc sa_d;
  auto v_d = tr.data();

  launchZero(&la, 0);

  count(v_d, &la, N);
  launchFinalize(&la);
  verify(&la);
  fill(v_d, &la, N);



  std::cout << la.size() << std::endl;
  imax = 0;
  ave = 0;
  z = 0;
  for (auto i = 0U; i < n; ++i) {
    auto x = la.size(i);
    if (x == 0) {
      z++;
      continue;
    }
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == la.size(n));
  std::cout << "found with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << z << std::endl;

  // now the inverse map (actually this is the direct....)
  AtomicPairCounter* dc_d;
  AtomicPairCounter dc(0);

  dc_d = &dc;
  fillBulk(dc_d, v_d, &la, N);
  finalizeBulk(dc_d, &la);
  verifyBulk(&la, dc_d);
  AtomicPairCounter sdc(0);
  fillBulk(&sdc, v_d, &sa_d, N);
  finalizeBulk(&sdc, &sa_d);
  verifyBulk(&sa_d, &sdc);

  std::cout << "final counter value " << dc.get().n << ' ' << dc.get().m << std::endl;

  std::cout << la.size() << std::endl;
  imax = 0;
  ave = 0;
  for (auto i = 0U; i < N; ++i) {
    auto x = la.size(i);
    if (!(x == 4 || x == 3))
      std::cout << i << ' ' << x << std::endl;
    assert(x == 4 || x == 3);
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == la.size(N));
  std::cout << "found with ave occupancy " << double(ave) / N << ' ' << imax << std::endl;

  // here verify use of block local counters
  auto m1_d = std::make_unique<Multiplicity>();
  auto m2_d = std::make_unique<Multiplicity>();
  launchZero(m1_d.get(), 0);
  launchZero(m2_d.get(), 0);

  countMulti(v_d, m1_d.get(), N);
  countMultiLocal(v_d, m2_d.get(), N);
  verifyMulti(m1_d.get(), m2_d.get());

  launchFinalize(m1_d.get());
  launchFinalize(m2_d.get());
  verifyMulti(m1_d.get(), m2_d.get());
  return 0;
}
