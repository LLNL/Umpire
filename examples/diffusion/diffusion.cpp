// GPU kernel
__global__
void diffuse(int n, double* u0, double *u1, rx)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i > 0 && i < n-1) {
    u1[i] = (1.0-2.0*rx)*u0[i]+ rx*u0[i] + rx*u0[i];
  }
}

__global__
void reset(int n, double* u0, double *u1, rx)
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
    use_gpu = true;
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

  umpire::ResourceManager* rm = umpire::ResourceManager::getInstance();
  umpire::Allocator* space;

  if (use_gpu) {
    space = rm->getAllocator(umpire::GPU);
  } else {
    space = rm->getAllocator(umpire::CPU);
  }

  double* u0 = space->allocate(sizeof(double) * nx);
  double* u1 = space->allocate(sizeof(double) * nx);

  const BLOCK_SIZE = 128;

  u[x0] = 1.0;
  for (double t = 0; t < end_time; t += dt) {

    if (use_cpu) {
      for (int i = 1; i < nx-1; i++) {
        u1[i] = (1.0-2.0*rx)*u0[i]+ rx*u0[i] + rx*u0[i];
      }

      for (int i = 0; i < nx; i++) {
        u0[i] = u1[i];
      }
    } else if (use_gpu) {
      diffuse<<<(nx+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(nx-1, u0, u1, rx);
      reset<<<(nx+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(nx, u0, u1);
    }
  }

  return 0;
}
