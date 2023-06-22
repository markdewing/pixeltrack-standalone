
// From plugin-SiPixelClusterizer/gpuClustering.h

#include "HistoContainer.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

const uint32_t maxNumModules = 2000;
const uint16_t InvId = 9999;

void findClus(uint16_t *id,
              uint16_t *x,
              uint16_t *y,
              uint32_t *moduleStart,
              uint32_t *nClustersInModule,  // output: number of clusters in each module
              uint32_t *moduleId,           // output: module id of each modules
              int32_t *clusterId,           // output: cluster id of each pixel
              int numElements) {
  uint32_t endModule = moduleStart[0];

  for (int i = 0; i < numElements; i++)
    clusterId[i] = i;

  for (uint32_t module = 0; module < endModule; module++) {
    auto firstPixel = moduleStart[module + 1];
    auto thisModuleId = id[firstPixel];
    //std::cout << "Module " << module << std::endl;

    // find the index of the first pixel not belonging to this module (or
    // invalid)
    int msize = numElements;

    // skip threads not associated to an existing pixel
    for (int i = firstPixel; i < numElements; i++) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (id[i] != thisModuleId) {  // find the first pixel in a different module
        msize = std::min(msize, i);
        break;
      }
    }

    // std::cout << module << " msize " << msize << " " << msize-firstPixel <<
    // std::endl;

    constexpr uint32_t maxPixInModule = 4000;
    constexpr auto nbins = 52 * 8 + 2;  // phase1PixelTopology::numColsInModule + 2

    using Hist = cms::HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;
    // T : type of discretized input values
    // NBINS : number of bins
    // SIZE : max number of element
    // S : number of significant bits in T
    // I : type stored in the container (usually an index in a vector of the
    // input values)
    Hist hist;

    for (uint32_t j = 0; j < Hist::totbins(); j++) {
      hist.off[j] = 0;
    }

    if (msize - firstPixel > maxPixInModule) {
      printf("too many pixels in module %d: %d > %d\n", thisModuleId, msize - firstPixel, maxPixInModule);
      msize = maxPixInModule + firstPixel;
    }

    assert((msize == numElements) or ((msize < numElements) and (id[msize] != thisModuleId)));

    uint32_t totGood = 0;
    // fill histo
    for (int i = firstPixel; i < msize; i++) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      hist.count(y[i]);
      totGood++;
    }

    hist.finalize();
    assert(hist.size() == totGood);

    for (int i = firstPixel; i < msize; i++) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      //std::cout << y[i] << " " << hist.bin(y[i]) << " " << hist.off[hist.bin(y[i])] << " " <<  i-firstPixel << std::endl;
      hist.fill(y[i], i - firstPixel);
    }

    //for (uint32_t j = 0; j < hist.size(); j++) {
    //       std::cout << module << " hist bin " << j << " " << hist.bins[j] <<
    //       std::endl;
    //}
    // for (uint32_t j = 0; j < Hist::totbins(); j++) {
    //       std::cout << module << " hist off " << j << " " << hist.off[j] <<
    //       std::endl;
    // }
    // std::cout << std::endl;

    // Can't do dynamic allocation on the device
    // auto maxiter = hist.size();
    constexpr int maxiter = 1024;

    // allocate space for duplicate pixels: a pixel can appear more than once
    // with different charge in the same event
    constexpr int maxNeighbours = 10;
    assert((hist.size() / 1) <= maxiter);
    // nearest neighbour
    uint16_t nn[maxiter][maxNeighbours];
    uint8_t nnn[maxiter];  // number of nn
    for (uint32_t k = 0; k < maxiter; ++k)
      nnn[k] = 0;

    // fill NN
    for (uint32_t j = 0, k = 0U; j < hist.size(); j++, ++k) {
      assert(k < maxiter);
      auto p = hist.begin() + j;
      auto i = *p + firstPixel;
      assert(id[i] != InvId);
      assert(id[i] == thisModuleId);  // same module
      int be = Hist::bin(y[i] + 1);
      auto e = hist.end(be);
      ++p;
      assert(0 == nnn[k]);
      for (; p < e; ++p) {
        auto m = (*p) + firstPixel;
        assert(m != i);
        assert(int(y[m]) - int(y[i]) >= 0);
        assert(int(y[m]) - int(y[i]) <= 1);
        // Compiling with LLVM, get nvlink error : Undefined reference to 'abs'
        // in
        // '/tmp/SiPixelRawToClusterGPUKernel.cc-nvptx64-nvidia-cuda-sm_61-4fd64f.cubin'
        // if (std::abs(int(x[m]) - int(x[i])) > 1)
        //  continue;
        if ((int(x[m]) - int(x[i])) > 1 || (int(x[m]) - int(x[i])) < -1)
          continue;

        auto l = nnn[k]++;
        assert(l < maxNeighbours);
        nn[k][l] = *p;
      }
    }
    //std::cout << "nnn" << std::endl;
    //for (uint32_t i = 0; i < hist.size(); i++) {
    //    std::cout << i << " " << (uint32_t)nnn[i] << std::endl;
    //}

    // for each pixel, look at all the pixels until the end of the module;
    // when two valid pixels within +/- 1 in x or y are found, set their id to
    // the minimum; after the loop, all the pixel in each cluster should have
    // the id equeal to the lowest pixel in the cluster ( clus[i] == i ).
    bool more = true;
    int nloops = 0;
    while (more) {
      if (1 == nloops % 2) {
        for (uint32_t j = 0, k = 0U; j < hist.size(); j++, ++k) {
          auto p = hist.begin() + j;
          auto i = *p + firstPixel;
          auto m = clusterId[i];
          while (m != clusterId[m])
            m = clusterId[m];
          clusterId[i] = m;
        }
      } else {
        more = false;
        for (uint32_t j = 0, k = 0U; j < hist.size(); j++, ++k) {
          auto p = hist.begin() + j;
          auto i = *p + firstPixel;
          for (int kk = 0; kk < nnn[k]; ++kk) {
            auto l = nn[k][kk];
            auto m = l + firstPixel;
            assert(m != i);
            // auto old = atomicMin(&clusterId[m], clusterId[i]);
            auto old = clusterId[m];
            clusterId[m] = std::min(clusterId[m], clusterId[i]);
            if (old != clusterId[i]) {
              // end the loop only if no changes were applied
              more = true;
            }
            // atomicMin(&clusterId[i], old);
            clusterId[i] = std::min(clusterId[i], old);
          }  // nnloop
        }    // pixel loop
      }
      ++nloops;
    }  // end while

    unsigned int foundClusters = 0;

    // find the number of different clusters, identified by a pixels with
    // clus[i] == i; mark these pixels with a negative id.
    for (int i = firstPixel; i < msize; i++) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (clusterId[i] == i) {
        // auto old = atomicInc(&foundClusters, 0xffffffff);
        auto old = foundClusters;
        foundClusters++;
        clusterId[i] = -(old + 1);
      }
    }

    // propagate the negative id to all the pixels in the cluster.
    for (int i = firstPixel; i < msize; i++) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (clusterId[i] >= 0) {
        // mark each pixel in a cluster with the same id as the first one
        clusterId[i] = clusterId[clusterId[i]];
      }
    }

    // adjust the cluster id to be a positive value starting from 0
    for (int i = firstPixel; i < msize; i++) {
      if (id[i] == InvId) {  // skip invalid pixels
        clusterId[i] = -9999;
        continue;
      }
      clusterId[i] = -clusterId[i] - 1;
    }

    nClustersInModule[thisModuleId] = foundClusters;
    moduleId[module] = thisModuleId;
  }
}

int main() {
  std::ifstream in("in_findclus1.txt");
  std::string line1;
  // in >> line1;
  getline(in, line1);  // modulesStart
  getline(in, line1);  // numModules
  const int numModules = 1789;
  // std::cout << "line1 = " << line1 << std::endl;

  const int numElements = 48316;
  // const int numElements = 4;

  uint32_t *moduleStart = new uint32_t[maxNumModules + 1];
  moduleStart[0] = numModules;

  for (int i = 0; i < numModules; i++) {
    int ii;
    int imodstart;
    in >> ii >> imodstart;
    // std::cout << ii  << " " << idi << std::endl;
    assert(ii == i + 1);
    moduleStart[i + 1] = imodstart;
  }

  uint16_t *id = new uint16_t[numElements];
  uint16_t *x = new uint16_t[numElements];
  uint16_t *y = new uint16_t[numElements];

  getline(in, line1);  // EOL
  getline(in, line1);  // blank
  getline(in, line1);  // numElements
  getline(in, line1);  // i id x y

  for (int i = 0; i < numElements; i++) {
    int ii;
    int idi;
    int ix;
    int iy;
    in >> ii >> idi >> ix >> iy;
    if (ii != i)
      std::cout << "about to assert, i = " << i << " ii = " << ii << std::endl;
    assert(ii == i);
    id[i] = idi;
    x[i] = ix;
    y[i] = iy;
  }

  getline(in, line1);  // EOL
  getline(in, line1);  // blank
  getline(in, line1);  // nClustersInModules moduleId

  uint32_t *refNClustersInModule = new uint32_t[maxNumModules];
  uint32_t *refModuleId = new uint32_t[maxNumModules];
  for (int i = 0; i < numModules; i++) {
    int ii;
    int nci;
    int modid;
    in >> ii >> nci >> modid;
    assert(ii == i);
    refNClustersInModule[i] = nci;
    refModuleId[i] = modid;
  }

  // getline(in, line1); // EOL
  // std::cout << "line1 = " << line1 << std::endl;
  getline(in, line1);  // blank
  getline(in, line1);  // clusterId

  int *refClusterId = new int[numElements];
  for (int i = 0; i < numElements; i++) {
    int ii;
    int cii;
    in >> ii >> cii;
    assert(ii == i);
    refClusterId[i] = cii;
  }

  uint32_t *nClustersInModule = new uint32_t[maxNumModules];
  uint32_t *moduleId = new uint32_t[maxNumModules];
  int *clusterId = new int[numElements];

  findClus(id, x, y, moduleStart, nClustersInModule, moduleId, clusterId, numElements);

  int errs = 0;
  for (int i = 0; i < numModules; i++) {
    if (nClustersInModule[i] != refNClustersInModule[i]) {
      std::cout << "clusters in module does not match " << i << " " << refNClustersInModule[i] << " "
                << nClustersInModule[i] << std::endl;
      errs++;
    }
    if (errs > 10)
      break;
  }

  for (int i = 0; i < numModules; i++) {
    if (moduleId[i] != refModuleId[i]) {
      std::cout << "module id does not match " << i << " " << refModuleId[i] << " " << moduleId[i] << std::endl;
      errs++;
    }

    if (errs > 10)
      break;
  }

  errs = 0;
  for (int i = 0; i < numElements; i++) {
    if (clusterId[i] != refClusterId[i]) {
      std::cout << "cluster id does not match " << i << " " << refClusterId[i] << " " << clusterId[i] << std::endl;
      errs++;
    }
    if (errs > 10)
      break;
  }

  if (errs == 0)
    std::cout << "Pass" << std::endl;
  else
    std::cout << "Fail" << std::endl;

  return 0;
}
