#include "gpuClusterTracksByDensity.h"
#include "gpuClusterTracksDBSCAN.h"
#include "gpuClusterTracksIterative.h"
#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"

namespace gpuVertexFinder {

  void loadTracks(TkSoA const* ptracks, ZVertexSoA* soa, WorkSpace* pws, float ptMin) {
    assert(ptracks);
    assert(soa);
    auto const& tracks = *ptracks;

    int nt = TkSoA::stride();
#pragma omp target teams distribute parallel for map(to:tracks,soa[:1],pws[:1])
    for (int idx = 0; idx < nt; idx++) {
      auto const& fit = tracks.stateAtBS;
      auto const* quality = tracks.qualityData();
      auto nHits = tracks.nHits(idx);
      if (nHits == 0)
        continue;
        //break;  // this is a guard: maybe we need to move to nTracks...


      // initialize soa...
      soa->idv[idx] = -1;

      if (nHits < 4)
        continue;  // no triplets
      if (quality[idx] != trackQuality::loose)
        continue;

      auto pt = tracks.pt(idx);

      if (pt < ptMin)
        continue;

      auto& data = *pws;
      //auto it = atomicAdd(&data.ntrks, 1);
      uint32_t it;
#pragma omp atomic capture
      it = data.ntrks++;

      data.itrk[it] = idx;
      data.zt[it] = tracks.zip(idx);
      data.ezt2[it] = fit.covariance(idx)(14);
      data.ptt2[it] = pt * pt;
    }
  }

// #define THREE_KERNELS
#ifndef THREE_KERNELS
  void vertexFinderOneKernel(gpuVertexFinder::ZVertices* pdata,
                             gpuVertexFinder::WorkSpace* pws,
                             int minT,      // min number of neighbours to be "seed"
                             float eps,     // max absolute distance to cluster
                             float errmax,  // max error to be "seed"
                             float chi2max  // max normalized distance to cluster,
  ) {
    clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max);

    fitVertices(pdata, pws, 50.);

    splitVertices(pdata, pws, 9.f);

    fitVertices(pdata, pws, 5000.);

    sortByPt2(pdata, pws);
  }
#else
  void vertexFinderKernel1(gpuVertexFinder::ZVertices* pdata,
                           gpuVertexFinder::WorkSpace* pws,
                           int minT,      // min number of neighbours to be "seed"
                           float eps,     // max absolute distance to cluster
                           float errmax,  // max error to be "seed"
                           float chi2max  // max normalized distance to cluster,
  ) {
    clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max);

    fitVertices(pdata, pws, 50.);
  }

  void vertexFinderKernel2(gpuVertexFinder::ZVertices* pdata, gpuVertexFinder::WorkSpace* pws) {
    fitVertices(pdata, pws, 5000.);

    sortByPt2(pdata, pws);
  }
#endif

  ZVertexHeterogeneous Producer::make(TkSoA const* tksoa, float ptMin) const {
    // std::cout << "producing Vertices on  CPU" <<    std::endl;
    ZVertexHeterogeneous vertices(std::make_unique<ZVertexSoA>());
    assert(tksoa);
    auto* soa = vertices.get();
    assert(soa);

    auto ws_d = std::make_unique<WorkSpace>();

    WorkSpace* pws = ws_d.get();
    init(soa, pws);
#pragma omp target enter data map(to:soa[:1],pws[:1])
    loadTracks(tksoa, soa, pws, ptMin);

    if (useDensity_) {
      clusterTracksByDensity(soa, pws, minT, eps, errmax, chi2max);
    } else if (useDBSCAN_) {
      clusterTracksDBSCAN(soa, pws, minT, eps, errmax, chi2max);
    } else if (useIterative_) {
      clusterTracksIterative(soa, pws, minT, eps, errmax, chi2max);
    }
    // std::cout << "found " << (*ws_d).nvIntermediate << " vertices " << std::endl;
    fitVertices(soa, pws, 50.);
    // one block per vertex!
    splitVertices(soa, pws, 9.f);
    fitVertices(soa, pws, 5000.);
#pragma omp target exit data map(from:soa[:1],pws[:1])
    sortByPt2(soa, pws);

    return vertices;
  }

}  // namespace gpuVertexFinder

#undef FROM
