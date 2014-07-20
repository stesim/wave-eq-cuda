#include "spdiamat.h"	// only required for definition of 'real' type

// use namespace to group the helper functions as CUDA doesn't allow device
// functions to be class members
namespace CudaHelper11
{

// calculate the first time step of the simulation
//    ip: size of vectors f,g and z in real2 entries
__global__
void calculateFirstStep( real dt, real h, real l2, real a, real b, int ip,
		const real2* f, const real2* g, real2* z )
{
	// shared memory containing (ip + 2) real2 entries
	extern __shared__ real2 shmem[];

	// calculate current subdomain's position in memory
	int id = blockIdx.y * gridDim.x + blockIdx.x;
	int idLast = gridDim.x * gridDim.y - 2;
	int vecIdx = id * ip;
	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	f += vecIdx;
	g += vecIdx;
	z += vecIdx;

	// shared memory offset by 1, so first real node has index 0
	real2* temp = (real2*)shmem + 1;

	// copy current solution to shared memory
	temp[ idx ] = f[ idx ];
	// set additional nodes' values to account for boundary conditions
	if( id == 0 )
	{
		if( idx == 0 )
		{
			temp[ -1 ].y = temp[ 0 ].y;
			temp[ ip ].x = 0.0;
		}
	}
	else if( id == idLast )
	{
		if( idx == ip - 1 )
		{
			temp[ -1 ].y = 0.0;
			temp[ ip ].x = temp[ ip - 1 ].x;
		}
	}
	else
	{
		if( idx == 0 )
		{
			temp[ -1 ].y = 0.0;
			temp[ ip ].x = 0.0;
		}
	}
	__syncthreads();

	// calculate first step
	real tmp3 = temp[ idx ].x;
	tmp3 *= tmp3 * tmp3;
	z[ idx ].x = 0.5 * l2 * ( temp[ idx - 1 ].y + temp[ idx ].y ) +
		a * temp[ idx ].x + dt * g[ idx ].x + b * tmp3;
	tmp3 = temp[ idx ].y;
	tmp3 *= tmp3 * tmp3;
	z[ idx ].y = 0.5 * l2 * ( temp[ idx ].x + temp[ idx + 1 ].x ) +
		a * temp[ idx ].y + dt * g[ idx ].y + b * tmp3;
}

// calculate a full cycle before rejoining the solutions - basically the same as
// calculating the first step except for a loop over the number of steps
__global__
void calculateNSteps( int nsteps, real l2, real a, real b, int ip,
		real2* z, real2* w )
{
	extern __shared__ real2 shmem[];

	int id = blockIdx.y * gridDim.x + blockIdx.x;
	int idLast = gridDim.x * gridDim.y - 2;
	int vecIdx = id * ip;
	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	z += vecIdx;
	w += vecIdx;

	real2* temp = (real2*)shmem + 1;

	real2* swap;
	for( int i = 0; i < nsteps; ++i )
	{
		__syncthreads();
		temp[ idx ] = z[ idx ];
		if( id == 0 )
		{
			if( idx == 0 )
			{
				temp[ -1 ].y = temp[ 0 ].y;
				temp[ ip ].x = 0.0;
			}
		}
		else if( id == idLast )
		{
			if( idx == ip - 1 )
			{
				temp[ -1 ].y = 0.0;
				temp[ ip ].x = temp[ ip - 1 ].x;
			}
		}
		else
		{
			if( idx == 0 && i == 0 )
			{
				temp[ -1 ].y = 0.0;
				temp[ ip ].x = 0.0;
			}
		}
		__syncthreads();

		real tmp3 = temp[ idx ].x;
		tmp3 *= tmp3 * tmp3;
		w[ idx ].x = l2 * ( temp[ idx - 1 ].y + temp[ idx ].y ) +
			a * temp[ idx ].x - w[ idx ].x + b * tmp3;
		tmp3 = temp[ idx ].y;
		tmp3 *= tmp3 * tmp3;
		w[ idx ].y = l2 * ( temp[ idx ].x + temp[ idx + 1 ].x ) +
			a * temp[ idx ].y - w[ idx ].y + b * tmp3;

		// swap vector pointers to avoid copy operations
		swap = w;
		w = z;
		z = swap;
	}
}

// replace invalid values with the correct ones by copying them from the
// neighboring subdomains
__global__
void synchronizeResults( int ip, real2* z, real2* w )
{
	int id = blockIdx.y * gridDim.x + blockIdx.x;
	int idLast = gridDim.x * gridDim.y - 2;
	int vecIdx = id * ip;
	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	z += vecIdx;
	w += vecIdx;

	// if not leftmost subdomain, copy from left neighbor
	if( id > 0 )
	{
		z[ idx ] = z[ idx - ip / 2 ];
		w[ idx ] = w[ idx - ip / 2 ];
	}

	idx += ip * 3 / 4;

	// if not rightmost, copy from right neighbor
	if( id < idLast )
	{
		z[ idx ] = z[ idx + ip / 2 ];
		w[ idx ] = w[ idx + ip / 2 ];
	}
}

}
