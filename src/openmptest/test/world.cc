#include <iostream>
#include <omp.h>


int main() {
  //std::cout << "World from" << std::endl;

  int ndevice = omp_get_num_devices();
  std::cout << "Number of devices = " << ndevice << std::endl;

  int default_device = omp_get_default_device();

  std::cout << "default device = " << default_device << std::endl;

  int is_host = -1;
#pragma omp target map(from:is_host)
  is_host = omp_is_initial_device();

  if (is_host == 0) {
    std::cout << "Offload region successfully ran on device" << std::endl;
  } else if (is_host == 1) {
    std::cout << "Offload region ran on host" << std::endl;
  } else {
    std::cout << "Offload region did not run, is_host = " << is_host << std::endl;
  }


  return 0;
}
