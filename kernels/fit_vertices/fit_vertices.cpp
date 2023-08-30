
// From plugin-PixelVertexFinding/fitVertices.h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <cassert>

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

void fitVertices(ZVertices* pdata,
                 WorkSpace* pws,
                 float chi2Max  // for outlier rejection
) {
  constexpr bool verbose = false;  // in principle the compiler should optmize out if false

  auto& __restrict__ data = *pdata;
  auto& __restrict__ ws = *pws;
  auto nt = ws.ntrks;
  float const* __restrict__ zt = ws.zt;
  float const* __restrict__ ezt2 = ws.ezt2;
  float* __restrict__ zv = data.zv;
  float* __restrict__ wv = data.wv;
  float* __restrict__ chi2 = data.chi2;
  uint32_t& nvFinal = data.nvFinal;
  uint32_t& nvIntermediate = ws.nvIntermediate;

  int32_t* __restrict__ nn = data.ndof;
  int32_t* __restrict__ iv = ws.iv;

  assert(pdata);
  assert(zt);

  assert(nvFinal <= nvIntermediate);
  nvFinal = nvIntermediate;
  auto foundClusters = nvFinal;

  // zero
  for (uint32_t i = 0; i < foundClusters; i++) {
    zv[i] = 0;
    wv[i] = 0;
    chi2[i] = 0;
  }

  // only for test
  int noise;
  if (verbose)
    noise = 0;

  // compute cluster location
  for (uint32_t i = 0; i < nt; i++) {
    if (iv[i] > 9990) {
      if (verbose)
        //atomicAdd(&noise, 1);
        noise++;
      continue;
    }
    assert(iv[i] >= 0);
    if (iv[i] >= foundClusters)
      std::cout << "iv[i] = " << iv[i] << " i = " << i << std::endl;
    assert(iv[i] < int(foundClusters));
    auto w = 1.f / ezt2[i];
    //atomicAdd(&zv[iv[i]], zt[i] * w);
    //atomicAdd(&wv[iv[i]], w);
    zv[iv[i]] += zt[i] * w;
    wv[iv[i]] += w;
  }

  // reuse nn
  for (uint32_t i = 0; i < foundClusters; i++) {
    if (!(wv[i] > 0.f))
      std::cout << "i = " << i << " wv[i] = " << wv[i] << std::endl;
    assert(wv[i] > 0.f);
    zv[i] /= wv[i];
    nn[i] = -1;  // ndof
  }

  // compute chi2
  for (uint32_t i = 0; i < nt; i++) {
    if (iv[i] > 9990)
      continue;

    auto c2 = zv[iv[i]] - zt[i];
    c2 *= c2 / ezt2[i];
    if (c2 > chi2Max) {
      iv[i] = 9999;
      continue;
    }
    //atomicAdd(&chi2[iv[i]], c2);
    //atomicAdd(&nn[iv[i]], 1);
    chi2[iv[i]] += c2;
    nn[iv[i]] += 1;
  }

  for (uint32_t i = 0; i < foundClusters; i++)
    if (nn[i] > 0)
      wv[i] *= float(nn[i]) / chi2[i];

  if (verbose)
    printf("found %d proto clusters ", foundClusters);
  if (verbose)
    printf("and %d noise\n", noise);
}

int main() {
  std::ifstream in("in_fit_vertices1.txt");
  std::string line1;
  std::string line2;

  getline(in, line1);  // fitVertices
  //std::cout << " line1 = " << line1 << std::endl;
  uint32_t foundClusters;
  uint32_t ntrks;
  in >> line1 >> foundClusters >> line1 >> ntrks;
  float chi2max;
  in >> line1 >> chi2max >> line1;

  getline(in, line1);  // # i iv zt ezt2
  WorkSpace ws;
  ws.ntrks = ntrks;
  ws.nvIntermediate = foundClusters;
  WorkSpace ws_out;
  ws_out.ntrks = ntrks;
  for (uint32_t i = 0; i < ntrks; i++) {
    int ii, iv;
    float zt, ezt2;
    in >> ii >> iv >> zt >> ezt2;
    assert(i == ii);
    ws.zt[i] = zt;
    ws.ezt2[i] = ezt2;
    ws.iv[i] = iv;
    ws_out.iv[i] = iv;
  }

  getline(in, line1);  // end of last line
  getline(in, line1);  // # i zv wv chi2 nn

  ZVertexSoA data;
  data.init();
  data.nvFinal = foundClusters;
  ZVertexSoA data_out;
  for (uint32_t i = 0; i < foundClusters; i++) {
    int ii;
    int nn;
    float zv, wv, chi2;
    in >> ii >> zv >> wv >> chi2 >> nn;
    assert(i == ii);
    data_out.ndof[i] = nn;
    data_out.zv[i] = zv;
    data_out.wv[i] = wv;
    data_out.chi2[i] = chi2;
  }

  //clusterTracksByDensity(&data, &ws, minT, eps, errmax, chi2max);
  fitVertices(&data, &ws, chi2max);

  int errs = 0;
  int all_errs = 0;
  for (uint32_t i = 0; i < ntrks; i++) {
    if (data.ndof[i] != data_out.ndof[i]) {
      std::cout << "ndof does not match " << i << " " << data.ndof[i] << " " << data_out.ndof[i] << std::endl;
      errs++;
    }
    if (errs > 10)
      break;
  }
  all_errs += errs;

  float tol = 0.0001;
  double zv_norm{0.0};
  errs = 0;
  for (uint32_t i = 0; i < ntrks; i++) {
    double diff = (data.zv[i] - data_out.zv[i]);
    double rel_diff = data_out.zv[i] > 0.0 ? diff / data_out.zv[i] : 0.0;
    zv_norm += rel_diff * rel_diff;
    if (std::abs(rel_diff) > tol) {
      std::cout << "zv does not match " << i << " " << data.zv[i] << " " << data_out.zv[i] << std::endl;
      errs++;
    }
    if (errs > 10)
      break;
  }
  std::cout << " relative norm of zv = " << std::sqrt(zv_norm) << std::endl;
  all_errs += errs;

  double wv_norm{0.0};
  errs = 0;
  for (uint32_t i = 0; i < ntrks; i++) {
    double diff = (data.wv[i] - data_out.wv[i]);
    double rel_diff = data_out.wv[i] > 0.0 ? diff / data_out.wv[i] : 0.0;
    wv_norm += rel_diff * rel_diff;
    if (std::abs(rel_diff) > tol) {
      std::cout << "wv does not match " << i << " " << data.wv[i] << " " << data_out.wv[i] << std::endl;
      errs++;
    }
    if (errs > 10)
      break;
  }
  std::cout << " relative norm of wv = " << std::sqrt(wv_norm) << std::endl;
  all_errs += errs;

  double chi2_norm{0.0};
  errs = 0;
  for (uint32_t i = 0; i < ntrks; i++) {
    double diff = (data.chi2[i] - data_out.chi2[i]);
    double rel_diff = data_out.chi2[i] > 0.0 ? diff / data_out.chi2[i] : 0.0;
    chi2_norm += rel_diff * rel_diff;
    if (std::abs(rel_diff) > tol) {
      std::cout << "chi2 does not match " << i << " " << data.chi2[i] << " " << data_out.chi2[i] << std::endl;
      errs++;
    }
    if (errs > 10)
      break;
  }
  std::cout << " relative norm of chi2 = " << std::sqrt(chi2_norm) << std::endl;

  all_errs += errs;
  if (all_errs == 0)
    std::cout << "Pass" << std::endl;
  else
    std::cout << "Fail" << std::endl;

  return 0;
}
