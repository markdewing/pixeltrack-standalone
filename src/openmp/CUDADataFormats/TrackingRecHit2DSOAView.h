#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h

#include "CUDACore/cudaCompat.h"

#include "CUDADataFormats/gpuClusteringConstants.h"
#include "CUDACore/HistoContainer.h"
#include "CUDACore/cudaCompat.h"
#include "Geometry/phase1PixelTopology.h"

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

class TrackingRecHit2DSOAView {
public:
  static constexpr uint32_t maxHits() { return gpuClustering::MaxNumClusters; }
  using hindex_type = uint16_t;  // if above is <=2^16

  using Hist =
      cms::cuda::HistoContainer<int16_t, 128, gpuClustering::MaxNumClusters, 8 * sizeof(int16_t), uint16_t, 10>;

  using AverageGeometry = phase1PixelTopology::AverageGeometry;

  template <typename>
  friend class TrackingRecHit2DHeterogeneous;

    uint32_t nHits() const { return m_nHits; }

    float& xLocal(int i) { return m_xl[i]; }
    float xLocal(int i) const { return m_xl[i]; }
    float& yLocal(int i) { return m_yl[i]; }
    float yLocal(int i) const { return m_yl[i]; }

    float& xerrLocal(int i) { return m_xerr[i]; }
    float xerrLocal(int i) const { return m_xerr[i]; }
    float& yerrLocal(int i) { return m_yerr[i]; }
    float yerrLocal(int i) const { return m_yerr[i]; }

    float& xGlobal(int i) { return m_xg[i]; }
    float xGlobal(int i) const { return m_xg[i]; }
    float& yGlobal(int i) { return m_yg[i]; }
    float yGlobal(int i) const { return m_yg[i]; }
    float& zGlobal(int i) { return m_zg[i]; }
    float zGlobal(int i) const { return m_zg[i]; }
    float& rGlobal(int i) { return m_rg[i]; }
    float rGlobal(int i) const { return m_rg[i]; }

    int16_t& iphi(int i) { return m_iphi[i]; }
    int16_t iphi(int i) const { return m_iphi[i]; }

    int32_t& charge(int i) { return m_charge[i]; }
    int32_t charge(int i) const { return m_charge[i]; }
    int16_t& clusterSizeX(int i) { return m_xsize[i]; }
    int16_t clusterSizeX(int i) const { return m_xsize[i]; }
    int16_t& clusterSizeY(int i) { return m_ysize[i]; }
    int16_t clusterSizeY(int i) const { return m_ysize[i]; }
    uint16_t& detectorIndex(int i) { return m_detInd[i]; }
    uint16_t detectorIndex(int i) const { return m_detInd[i]; }

    pixelCPEforGPU::ParamsOnGPU const& cpeParams() const { return *m_cpeParams; }

    uint32_t hitsModuleStart(int i) const { return m_hitsModuleStart[i]; }

    uint32_t* hitsLayerStart() { return m_hitsLayerStart; }
    uint32_t const* hitsLayerStart() const { return m_hitsLayerStart; }

    Hist& phiBinner() { return *m_hist; }
    Hist const& phiBinner() const { return *m_hist; }

    AverageGeometry& averageGeometry() { return *m_averageGeometry; }
    AverageGeometry const& averageGeometry() const { return *m_averageGeometry; }

private:
  // local coord
  float *m_xl, *m_yl;
  float *m_xerr, *m_yerr;

  // global coord
  float *m_xg, *m_yg, *m_zg, *m_rg;
  int16_t* m_iphi;

  // cluster properties
  int32_t* m_charge;
  int16_t* m_xsize;
  int16_t* m_ysize;
  uint16_t* m_detInd;

  // supporting objects
  AverageGeometry* m_averageGeometry;  // owned (corrected for beam spot: not sure where to host it otherwise)
  pixelCPEforGPU::ParamsOnGPU const* m_cpeParams;  // forwarded from setup, NOT owned
  uint32_t const* m_hitsModuleStart;               // forwarded from clusters

  uint32_t* m_hitsLayerStart;

  Hist* m_hist;

  uint32_t m_nHits;
};

#endif
