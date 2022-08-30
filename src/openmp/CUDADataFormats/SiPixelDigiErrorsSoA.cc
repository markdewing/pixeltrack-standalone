#include "CUDADataFormats/SiPixelDigiErrorsSoA.h"

#include <cassert>
#include <cstring>

SiPixelDigiErrorsSoA::SiPixelDigiErrorsSoA(size_t maxFedWords, PixelFormatterErrors errors)
    : formatterErrors_h(std::move(errors)) {
  error_d = std::make_unique<cms::openmp::SimpleVector<PixelErrorCompact>>();
  data_d = std::make_unique<PixelErrorCompact[]>(maxFedWords);

  std::memset(data_d.get(), 0x00, maxFedWords);

  error_d = std::make_unique<cms::openmp::SimpleVector<PixelErrorCompact>>();
  cms::openmp::make_SimpleVector(error_d.get(), maxFedWords, data_d.get());
  assert(error_d->empty());
  assert(error_d->capacity() == static_cast<int>(maxFedWords));
}
