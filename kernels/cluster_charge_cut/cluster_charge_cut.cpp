
#include <iostream>
#include <fstream>
#include <cassert>
#include "prefixScan.h"

const uint32_t maxNumModules = 2000;
const uint16_t InvId = 9999;
constexpr int32_t MaxNumClustersPerModules = 1024;

void clusterChargeCut(uint16_t* id,                 // module id of each pixel (modified if bad cluster)
                      uint16_t const* adc,          // charge of each pixel
                      uint32_t const* moduleStart,  // index of the first pixel of each module
                      uint32_t* nClustersInModule,  // modified: number of clusters found in each module
                      uint32_t const* moduleId,     // module id of each module
                      int32_t* clusterId,           // modified: cluster id each pixel
                      uint32_t numElements) {
  int32_t charge[MaxNumClustersPerModules];
  uint8_t ok[MaxNumClustersPerModules];
  uint16_t newclusId[MaxNumClustersPerModules];
  auto endModule = moduleStart[0];
  for (auto module = 0; module < endModule; module++) {
    auto firstPixel = moduleStart[1 + module];
    auto thisModuleId = id[firstPixel];
    assert(thisModuleId < maxNumModules);
    // The presence of this assert will cause wrong values in nClustersInModule for LLVM 15
    //assert(thisModuleId == moduleId[module]);

    auto nclus = nClustersInModule[thisModuleId];
    if (nclus == 0)
      continue;

    if (nclus > MaxNumClustersPerModules)
      printf("Warning too many clusters in module %d in block %d: %d > %d\n",
             thisModuleId,
             0,
             nclus,
             MaxNumClustersPerModules);

    auto first = firstPixel;

    if (nclus > MaxNumClustersPerModules) {
      // remove excess  FIXME find a way to cut charge first....
      for (auto i = first; i < numElements; i++) {
        if (id[i] == InvId)
          continue;  // not valid
        if (id[i] != thisModuleId)
          break;  // end of module
        if (clusterId[i] >= MaxNumClustersPerModules) {
          id[i] = InvId;
          clusterId[i] = InvId;
        }
      }
      nclus = MaxNumClustersPerModules;
    }

    assert(nclus <= MaxNumClustersPerModules);
    for (uint32_t i = 0; i < nclus; i++) {
      charge[i] = 0;
    }

    for (auto i = first; i < numElements; i++) {
      if (id[i] == InvId)
        continue;  // not valid
      if (id[i] != thisModuleId)
        break;  // end of module
      charge[clusterId[i]] += adc[i];
    }

    auto chargeCut = thisModuleId < 96 ? 2000 : 4000;  // move in constants (calib?)
    for (uint32_t i = 0; i < nclus; i++) {
      newclusId[i] = ok[i] = charge[i] > chargeCut ? 1 : 0;
    }

    cms::cuda::blockPrefixScan(newclusId, nclus);

    assert(nclus >= newclusId[nclus - 1]);

    if (nclus == newclusId[nclus - 1])
      continue;

    nClustersInModule[thisModuleId] = newclusId[nclus - 1];

    // mark bad cluster again
    for (uint32_t i = 0; i < nclus; i++) {
      if (0 == ok[i])
        newclusId[i] = InvId + 1;
    }

    // reassign id
    for (auto i = first; i < numElements; i++) {
      if (id[i] == InvId)
        continue;  // not valid
      if (id[i] != thisModuleId)
        break;  // end of module
      clusterId[i] = newclusId[clusterId[i]] - 1;
      if (clusterId[i] == InvId)
        id[i] = InvId;
    }
  }
}

int main() {
  std::ifstream in("in_cluster_charge_cut1.txt");
  std::string line1;

  getline(in, line1);  // clusterChargeCut
  getline(in, line1);  // numModules
  getline(in, line1);  //  i moduleStart[i+1]  moduleid[i]  nClustersInModule[i]
  const int numModules = 1789;

  const int numElements = 48316;

  uint32_t* moduleStart = new uint32_t[maxNumModules + 1];
  uint32_t* moduleId = new uint32_t[maxNumModules];
  uint32_t* nClustersInModule = new uint32_t[maxNumModules];
  moduleStart[0] = numModules;

  for (int i = 0; i < numModules; i++) {
    int ii;
    int imodstart;
    int imodid;
    int i_nclus;
    in >> ii >> imodstart >> imodid >> i_nclus;
    // std::cout << ii  << " " << idi << std::endl;
    assert(ii == i);
    moduleStart[i + 1] = imodstart;
    moduleId[i] = imodid;
    nClustersInModule[i] = i_nclus;
  }
  getline(in, line1);  // EOL
  getline(in, line1);  // blank
  getline(in, line1);  // numElements
  getline(in, line1);  // i id adc clusterId

  uint16_t* id = new uint16_t[numElements];
  uint16_t* adc = new uint16_t[numElements];
  int32_t* clusterId = new int32_t[numElements];
  for (int i = 0; i < numElements; i++) {
    int ii;
    int iid;
    int iadc;
    int iclusid;
    in >> ii >> iid >> iadc >> iclusid;
    assert(ii == i);
    id[i] = iid;
    adc[i] = iadc;
    clusterId[i] = iclusid;
  }

  getline(in, line1);  // EOL
  getline(in, line1);  // blank
  getline(in, line1);  // blank
  getline(in, line1);  // Output
  getline(in, line1);  // i nClustersInModule[i]

  uint32_t* refNClustersInModule = new uint32_t[maxNumModules];
  for (int i = 0; i < numModules; i++) {
    int ii;
    int i_ncls;
    in >> ii >> i_ncls;
    assert(ii == i);
    refNClustersInModule[i] = i_ncls;
  }

  getline(in, line1);  // EOL
  getline(in, line1);  // i id clusterId
                       //
  uint16_t* refId = new uint16_t[numElements];
  int32_t* refClusterId = new int32_t[numElements];
  for (int i = 0; i < numElements; i++) {
    int ii;
    int iid;
    int iclusid;
    in >> ii >> iid >> iclusid;
    assert(ii == i);
    refId[i] = iid;
    refClusterId[i] = iclusid;
  }

  clusterChargeCut(id, adc, moduleStart, nClustersInModule, moduleId, clusterId, numElements);

  int errs = 0;
  for (int i = 0; i < numModules; i++) {
    if (nClustersInModule[i] != refNClustersInModule[i]) {
      std::cout << "nclustersInModuleId does not match " << i << " " << refNClustersInModule[i] << " "
                << nClustersInModule[i] << std::endl;
      errs++;
    }
    if (errs > 0)
      break;
  }

  errs = 0;
  for (int i = 0; i < numElements; i++) {
    if (id[i] != refId[i]) {
      std::cout << "id does not match " << i << " " << refId[i] << " " << id[i] << std::endl;
      errs++;
    }
    if (errs > 10)
      break;
  }

  errs = 0;
  for (int i = 0; i < numElements; i++) {
    if (clusterId[i] != refClusterId[i]) {
      std::cout << "clusterId does not match " << i << " " << refClusterId[i] << " " << clusterId[i] << std::endl;
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
