#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

__global__
void init(int n, double* u0, int hotspot)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i > 0 && i < n-1) {
    u0[i] = 0.0;
  }

  if (i == hotspot) {
    u0[i] = 1.0;
  }
}

__global__
void diffuse(int n, double* u0, double *u1, double rx)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i > 0 && i < n-1) {
    u1[i] = (1.0-2.0*rx)*u0[i]+ rx*u0[i] + rx*u0[i];
  }
}

__global__
void reset(int n, double* u0, double *u1)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < n) 
    u0[i] = u1[i];
}

int main(int argc, char* argv[]) {
  int nx = 200;
  int steps = 10;

  bool use_gpu = false;

  if (atoi(argv[1]) == 1) {
    std::cout << "Running on GPU" << std::endl;
    use_gpu = true;
  } else {
    std::cout << "Running on GPU" << std::endl;
  }

  // location of point heat source
  int hotspot = (nx - 1) / 2;

  // constants used in the solution
  const double dx = 2.0 / (nx - 1);
  const double dt = 0.4 * (dx * dx);
  const double end_time = steps * dt;
  const double rx = dt/(dx*dx);

  if (rx > 0.5) {
    std::cerr << "Error, k/h^2 > 0.5!" << std::endl;
  }

  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator* space;

  if (use_gpu) {
    space = new umpire::Allocator(rm.getAllocator("DEVICE"));
  } else {
    space = new umpire::Allocator(rm.getAllocator("HOST"));
  }

  double* u0 = (double*)space->allocate(sizeof(double) * nx);
  double* u1 = (double*)space->allocate(sizeof(double) * nx);

  const int BLOCK_SIZE = 128;


  if (!use_gpu) {
    for (int i = 0; i < nx; i++) {
      u0[i] = 0.0;
    }
    u0[hotspot] = 1.0;

    std::cout << "Value at u0[ " << hotspot << "] = " << u0[hotspot] << std::endl;
  } else {
    init<<<(nx+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(nx, u0, hotspot);
  }

  for (double t = 0; t < end_time; t += dt) {
    std::cout << "Starting step " << t/dt << "....";

    if (!use_gpu) {
      for (int i = 1; i < nx-1; i++) {
        u1[i] = (1.0-2.0*rx)*u0[i]+ rx*u0[i-1] + rx*u0[i+1];
      }

      for (int i = 0; i < nx; i++) {
        u0[i] = u1[i];
      }
    } else if (use_gpu) {
      diffuse<<<(nx+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(nx-1, u0, u1, rx);
      reset<<<(nx+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(nx, u0, u1);
    }

    std::cout << "done." << std::endl;
  }


  if (!use_gpu) {
    std::cout << "Value at u0[ " << hotspot << "] = " << u0[hotspot] << std::endl;
  }

  space->deallocate(u0);
  space->deallocate(u1);

  return 0;
}
