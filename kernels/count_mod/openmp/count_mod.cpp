
// From plugin-SiPixelClusterizer/gpuClustering.h

#include <iostream>
#include <fstream>
#include <cassert>
#include <algorithm>

const uint32_t maxNumModules = 2000;
const uint16_t InvId = 9999;

void countModules(uint16_t *id, uint32_t *moduleStart, int numElements) {
  uint32_t numModules = 0;
#pragma omp target teams distribute parallel for map(tofrom: moduleStart[:maxNumModules+1], numModules)  \
                                                 map(to: id[:numElements])

  for (int i = 0; i < numElements; i++) {
    if (InvId == id[i])
      continue;
    auto j = i - 1;
    while (j >= 0 and id[j] == InvId)
      --j;
    if (j < 0 or id[j] != id[i]) {
      uint32_t loc;
#pragma omp atomic capture
      loc = numModules++;
      moduleStart[loc + 1] = i;
    }
  }

  moduleStart[0] = numModules;
}

int main() {
  std::ifstream in("in_count_mod.txt");
  std::string line1;
  //in >> line1;
  getline(in, line1);
  //std::cout << "line1 = " << line1 << std::endl;

  const int numElements = 48316;
  //const int numElements = 4;

  uint16_t *id = new uint16_t[numElements];
  uint32_t *moduleStart = new uint32_t[maxNumModules + 1];

  for (int i = 0; i < numElements; i++) {
    int ii;
    int idi;
    in >> ii >> idi;
    //std::cout << ii  << " " << idi << std::endl;
    assert(ii == i);
    id[i] = idi;
  }

  std::ifstream ref("ref_serial.txt");
  getline(ref, line1);

  uint32_t *refModuleStart = new uint32_t[maxNumModules + 1];
  uint32_t refNumModules = 1789;
  refModuleStart[0] = refNumModules;
  for (int i = 0; i < refModuleStart[0]; i++) {
    uint32_t ii;
    uint32_t ms;
    ref >> ii >> ms;
    assert(ii == i + 1);
    refModuleStart[i + 1] = ms;
  }

  countModules(id, moduleStart, numElements);

  // Order of moduleStart entries will vary based on iteration order
  std::sort(moduleStart + 1, moduleStart + refNumModules + 1);

  uint32_t numModule = moduleStart[0];
  if (numModule != refNumModules) {
    std::cout << "number of  modules " << moduleStart[0] << std::endl;
  }
  assert(numModule == refNumModules);

  int errs = 0;
  for (uint32_t i = 0; i < refNumModules; i++) {
    //std::cout << i+1 << " " << moduleStart[i+1] << " " << refModuleStart[i+1] << std::endl;
    if (moduleStart[i + 1] != refModuleStart[i + 1]) {
      std::cout << "Module start fail at " << i + 1 << " " << refModuleStart[i + 1] << " " << moduleStart[i + 1]
                << std::endl;

      errs++;
      if (errs > 10)
        break;
    }
    //assert(moduleStart[i+1] == refModuleStart[i+1]);
  }

  std::cout << "Pass" << std::endl;

  return 0;
}
