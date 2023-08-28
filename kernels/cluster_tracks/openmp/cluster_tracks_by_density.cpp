
// From plugin-PixelVertexFinding/gpuClusterTracksByDensity.h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>

#include "HistoContainer.h"

struct ZVertexSoA {
  static constexpr uint32_t MAXTRACKS = 32 * 1024;
  static constexpr uint32_t MAXVTX = 1024;

  int16_t idv[MAXTRACKS];    // vertex index for each associated (original) track  (-1 == not associate)
  float zv[MAXVTX];          // output z-posistion of found vertices
  float wv[MAXVTX];          // output weight (1/error^2) on the above
  float chi2[MAXVTX];        // vertices chi2
  float ptv2[MAXVTX];        // vertices pt^2
  int32_t ndof[MAXTRACKS];   // vertices number of dof (reused as workspace for the number of nearest neighbours FIXME)
  uint16_t sortInd[MAXVTX];  // sorted index (by pt2)  ascending
  uint32_t nvFinal;          // the number of vertices

  void init() { nvFinal = 0; }
};

using ZVertices = ZVertexSoA;

// workspace used in the vertex reco algos
struct WorkSpace {
  static constexpr uint32_t MAXTRACKS = ZVertexSoA::MAXTRACKS;
  static constexpr uint32_t MAXVTX = ZVertexSoA::MAXVTX;

  uint32_t ntrks;            // number of "selected tracks"
  uint16_t itrk[MAXTRACKS];  // index of original track
  float zt[MAXTRACKS];       // input track z at bs
  float ezt2[MAXTRACKS];     // input error^2 on the above
  float ptt2[MAXTRACKS];     // input pt^2 on the above
  uint8_t izt[MAXTRACKS];    // interized z-position of input tracks
  int32_t iv[MAXTRACKS];     // vertex index for each associated track

  uint32_t nvIntermediate;  // the number of vertices after splitting pruning etc.

  void init() {
    ntrks = 0;
    nvIntermediate = 0;
  }
};

// this algo does not really scale as it works in a single block...
// enough for <10K tracks we have
//
// based on Rodrighez&Laio algo
//
void clusterTracksByDensity(ZVertices* pdata,
                            WorkSpace* pws,
                            int minT,      // min number of neighbours to be "seed"
                            float eps,     // max absolute distance to cluster
                            float errmax,  // max error to be "seed"
                            float chi2max  // max normalized distance to cluster
) {
  constexpr bool verbose = false;  // in principle the compiler should optmize out if false

  constexpr uint32_t MAXTRACKS = WorkSpace::MAXTRACKS;
  if (verbose)
    printf("params %d %f %f %f\n", minT, eps, errmax, chi2max);

  auto er2mx = errmax * errmax;

  auto& __restrict__ data = *pdata;
  auto& __restrict__ ws = *pws;
  auto nt = ws.ntrks;
  float const* __restrict__ zt = ws.zt;
  float const* __restrict__ ezt2 = ws.ezt2;

  uint32_t& nvFinal = data.nvFinal;
  uint32_t& nvIntermediate = ws.nvIntermediate;

  uint8_t* __restrict__ izt = ws.izt;
  int32_t* __restrict__ nn = data.ndof;
  int32_t* __restrict__ iv = ws.iv;

  assert(pdata);
  assert(zt);

  using Hist = cms::HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;
  Hist hist;

  //#pragma omp target enter data map(to: zt[:MAXTRACKS], ezt2[:MAXTRACKS], izt[:MAXTRACKS], nn[:MAXTRACKS]) \
//                          map(alloc: iv[:MAXTRACKS], hist)

#pragma omp target teams distribute parallel for
  for (uint32_t j = 0; j < Hist::totbins(); j++) {
    hist.off[j] = 0;
  }

  if (verbose)
    printf("booked hist with %d bins, size %d for %d tracks\n", hist.nbins(), hist.capacity(), nt);

  assert(nt <= hist.capacity());

  //Libomptarget device 0 info: firstprivate(nt)[4] (implicit)
  //Libomptarget device 0 info: use_address(zt)[0] (implicit)
  //Libomptarget device 0 info: use_address(izt)[0] (implicit)
  //Libomptarget device 0 info: tofrom(hist)[33032] (implicit)
  //Libomptarget device 0 info: use_address(iv)[0] (implicit)
  //Libomptarget device 0 info: use_address(nn)[0] (implicit)

  // fill hist  (bin shall be wider than "eps")
#pragma omp target teams distribute parallel for map( \
        tofrom : zt[ : MAXTRACKS], izt[ : MAXTRACKS], iv[ : MAXTRACKS], nn[ : MAXTRACKS])
  for (uint32_t i = 0; i < nt; i++) {
    assert(i < ZVertices::MAXTRACKS);
    int iz = int(zt[i] * 10.);  // valid if eps<=0.1
    // iz = std::clamp(iz, INT8_MIN, INT8_MAX);  // sorry c++17 only
    iz = std::min(std::max(iz, INT8_MIN), INT8_MAX);
    izt[i] = iz - INT8_MIN;
    assert(iz - INT8_MIN >= 0);
    assert(iz - INT8_MIN < 256);
    hist.count(izt[i]);
    iv[i] = i;
    nn[i] = 0;
  }
  //#pragma omp target exit data map(delete: zt[:MAXTRACKS],ezt2[:MAXTRACKS],izt[:MAXTRACKS])  map(from:iv[:MAXTRACKS],nn[:MAXTRACKS])

#pragma omp target
  hist.finalize();

#pragma omp target update from(hist)
  assert(hist.size() == nt);
#pragma omp target teams distribute parallel for map(to : izt[ : MAXTRACKS])
  for (uint32_t i = 0; i < nt; i++) {
    hist.fill(izt[i], uint16_t(i));
  }

  // count neighbours
#pragma omp target teams distribute parallel for map( \
        tofrom : ezt2[ : MAXTRACKS], zt[ : MAXTRACKS], nn[ : MAXTRACKS], izt[ : MAXTRACKS])
  for (uint32_t i = 0; i < nt; i++) {
    if (ezt2[i] > er2mx)
      continue;
    auto loop = [&](uint32_t j) {
      if (i == j)
        return;
      auto dist = std::abs(zt[i] - zt[j]);
      if (dist > eps)
        return;
      if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
        return;
      nn[i]++;
    };

    cms::forEachInBins(hist, izt[i], 1, loop);
  }

  // find closest above me .... (we ignore the possibility of two j at same distance from i)
#pragma omp target teams distribute parallel for map( \
        tofrom : ezt2[ : MAXTRACKS], zt[ : MAXTRACKS], nn[ : MAXTRACKS], izt[ : MAXTRACKS], iv[ : MAXTRACKS])
  for (uint32_t i = 0; i < nt; i++) {
    float mdist = eps;
    auto loop = [&](uint32_t j) {
      if (nn[j] < nn[i])
        return;
      if (nn[j] == nn[i] && zt[j] >= zt[i])
        return;  // if equal use natural order...
      auto dist = std::abs(zt[i] - zt[j]);
      if (dist > mdist)
        return;
      if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
        return;  // (break natural order???)
      mdist = dist;
      iv[i] = j;  // assign to cluster (better be unique??)
    };
    cms::forEachInBins(hist, izt[i], 1, loop);
  }

#ifdef GPU_DEBUG
#pragma omp target teams distribute parallel for map(to : iv[MAXTRACKS])
  //  mini verification
  for (uint32_t i = 0; i < nt; i++) {
    if (iv[i] != int(i))
      assert(iv[iv[i]] != int(i));
  }

#endif

  // consolidate graph (percolate index of seed)
#pragma omp target teams distribute parallel for map(tofrom : iv[ : MAXTRACKS])
  for (uint32_t i = 0; i < nt; i++) {
    auto m = iv[i];
    while (m != iv[m])
      m = iv[m];
    iv[i] = m;
  }

#ifdef GPU_DEBUG
#pragma omp target teams distribute parallel for map(to : iv[ : MAXTRACKS])
  //  mini verification
  for (uint32_t i = 0; i < nt; i++) {
    if (iv[i] != int(i))
      assert(iv[iv[i]] != int(i));
  }
#endif

#ifdef GPU_DEBUG
  // and verify that we did not spit any cluster...
#pragma omp target teams distribute parallel for map( \
        to : nn[ : MAXTRACKS], zt[ : MAXTRACKS], ezt2[ : MAXTRACKS], iv[ : MAXTRACKS], izt[ : MAXTRACKS])
  for (uint32_t i = 0; i < nt; i++) {
    auto minJ = i;
    auto mdist = eps;
    auto loop = [&](uint32_t j) {
      if (nn[j] < nn[i])
        return;
      if (nn[j] == nn[i] && zt[j] >= zt[i])
        return;  // if equal use natural order...
      auto dist = std::abs(zt[i] - zt[j]);
      if (dist > mdist)
        return;
      if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
        return;
      mdist = dist;
      minJ = j;
    };
    cms::forEachInBins(hist, izt[i], 1, loop);
    // should belong to the same cluster...
    assert(iv[i] == iv[minJ]);
    assert(nn[i] <= nn[iv[i]]);
  }

#endif

  unsigned int foundClusters = 0;

  // find the number of different clusters, identified by a tracks with clus[i] == i and density larger than threshold;
  // mark these tracks with a negative id.
// Test fails validation with LLVM mainline if this loop is offloaded
#pragma omp target teams distribute parallel for map(tofrom : iv[ : MAXTRACKS], nn[ : MAXTRACKS], foundClusters)
  for (uint32_t i = 0; i < nt; i++) {
    if (iv[i] == int(i)) {
      if (nn[i] >= minT) {
        //auto old = atomicInc(&foundClusters, 0xffffffff);
        uint32_t old;
#pragma omp atomic capture
        old = foundClusters++;
        iv[i] = -(old + 1);
      } else {  // noise
        iv[i] = -9998;
      }
    }
  }

  assert(foundClusters < ZVertices::MAXVTX);

  // propagate the negative id to all the tracks in the cluster.
#pragma omp target teams distribute parallel for map(tofrom : iv[ : MAXTRACKS])
  for (uint32_t i = 0; i < nt; i++) {
    if (iv[i] >= 0) {
      // mark each track in a cluster with the same id as the first one
      iv[i] = iv[iv[i]];
    }
  }

  // adjust the cluster id to be a positive value starting from 0
#pragma omp target teams distribute parallel for map(tofrom : iv[ : MAXTRACKS])
  for (uint32_t i = 0; i < nt; i++) {
    iv[i] = -iv[i] - 1;
  }

  nvIntermediate = nvFinal = foundClusters;

  if (verbose)
    printf("found %d proto vertices\n", foundClusters);
}

int main() {
  std::ifstream in("in_cluster_tracks1.txt");
  std::string line1;
  std::string line2;

  getline(in, line1);  // clusterTracksByDensity
  std::cout << " line1 = " << line1 << std::endl;
  int minT;
  float eps;
  float errmax;
  float chi2max;
  in >> line1 >> minT >> line1 >> eps >> line1 >> errmax >> line1 >> chi2max >> line1;
  std::cout << " minT = " << minT << " eps = " << eps << " errmax " << errmax << " chi2max = " << chi2max << std::endl;
  getline(in, line1);  // MAXTRACKS
  getline(in, line1);  // MAXVTX
  uint32_t ntrks;
  in >> line1 >> line2 >> ntrks;  // "Workspace in" number of tracks
  std::cout << ntrks << std::endl;

  WorkSpace ws;
  ws.ntrks = ntrks;
  for (uint32_t i = 0; i < ntrks; i++) {
    int ii;
    int itrk, izt;
    float zt, ezt2, ptt2, iv;
    in >> ii >> itrk >> zt >> ezt2 >> ptt2 >> izt >> iv;
    assert(i == ii);
    ws.itrk[i] = itrk;
    ws.zt[i] = zt;
    ws.ezt2[i] = ezt2;
    ws.ptt2[i] = ptt2;
    ws.izt[i] = izt;
    ws.iv[i] = iv;
  }

  int foundClusters;
  in >> line1 >> foundClusters;  // ZVertices

  ZVertexSoA data;
  data.init();
  ZVertexSoA data_out;
  for (uint32_t i = 0; i < ntrks; i++) {
    int ii;
    int nn;
    in >> ii >> nn;
    assert(i == ii);
    data_out.ndof[i] = nn;
  }
  WorkSpace ws_out;
  int ntrks_out;
  in >> line1 >> line1 >> ntrks_out;  // "Workspace out" "ntrks"
  ws_out.ntrks = ntrks_out;
  for (uint32_t i = 0; i < ntrks_out; i++) {
    int ii;
    int itrk, izt;
    float zt, ezt2, ptt2, iv;
    in >> ii >> itrk >> zt >> ezt2 >> ptt2 >> izt >> iv;
    assert(i == ii);
    ws_out.itrk[i] = itrk;
    ws_out.zt[i] = zt;
    ws_out.ezt2[i] = ezt2;
    ws_out.ptt2[i] = ptt2;
    ws_out.izt[i] = izt;
    ws_out.iv[i] = iv;
  }

  clusterTracksByDensity(&data, &ws, minT, eps, errmax, chi2max);

  if (ws.ntrks != ws_out.ntrks)
    std::cout << "ws.ntrks = " << ws.ntrks << " ws_out.ntrks = " << ws_out.ntrks << std::endl;

  int errs = 0;
  int all_errs = 0;
  for (uint32_t i = 0; i < ntrks_out; i++) {
    if (data.ndof[i] != data_out.ndof[i]) {
      std::cout << "ndof does not match " << i << " " << data.ndof[i] << " " << data_out.ndof[i] << std::endl;
      errs++;
    }
    if (errs > 10)
      break;
  }
  all_errs += errs;

  errs = 0;
  for (uint32_t i = 0; i < ntrks_out; i++) {
    if (ws.iv[i] != ws_out.iv[i]) {
      std::cout << "iv does not match " << i << " " << ws.iv[i] << " " << ws_out.iv[i] << std::endl;
      errs++;
    }
    if (errs > 10)
      break;
  }
  all_errs += errs;

  if (all_errs == 0)
    std::cout << "Pass" << std::endl;
  else
    std::cout << "Fail" << std::endl;

  return 0;
}
