/* Component-wise + hardcoded matrix */

#include <stdio.h>
#include "spdiamat.h"

namespace CudaHelper11
{

__global__
void calculateFirstStep(
		real dt,
		real h,
		int ip,
		const real2* f,
		const real2* g,
		real2* z )
{
	extern __shared__ real2 shmem[];

	int id = blockIdx.y * gridDim.x + blockIdx.x;
	int idLast = gridDim.x * gridDim.y - 2;
	int vecIdx = id * ip;
	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	f += vecIdx;
	g += vecIdx;
	z += vecIdx;

	real2* temp = (real2*)shmem + 1;

	real l2 = dt / h * dt / h;
	real a = 1.0 - l2;

	temp[ idx ].x = f[ idx ].x;
	temp[ idx ].y = f[ idx ].y;
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

	z[ idx ].x = 0.5 * l2 * ( temp[ (int)idx - 1 ].y + temp[ idx ].y ) + a * temp[ idx ].x + dt * g[ idx ].x;
	z[ idx ].y = 0.5 * l2 * ( temp[ idx ].x + temp[ idx + 1 ].x ) + a * temp[ idx ].y + dt * g[ idx ].y;
}

__global__
void calculateNSteps(
		int nsteps,
		real l2,
		real a,
		int ip,
		real2* z,
		real2* w )
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
		temp[ idx ].x = z[ idx ].x;
		temp[ idx ].y = z[ idx ].y;
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

		w[ idx ].x = l2 * ( temp[ (int)idx - 1 ].y + temp[ idx ].y ) + a * temp[ idx ].x - w[ idx ].x;
		w[ idx ].y = l2 * ( temp[ idx ].x + temp[ idx + 1 ].x ) + a * temp[ idx ].y - w[ idx ].y;

		swap = w;
		w = z;
		z = swap;
	}
}

__global__
void synchronizeResults( int ip, real2* z, real2* w )
{
	int id = blockIdx.y * gridDim.x + blockIdx.x;
	int idLast = gridDim.x * gridDim.y - 2;
	int vecIdx = id * ip;
	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	z += vecIdx;
	w += vecIdx;

	if( id > 0 )
	{
		z[ idx ] = z[ idx - ip / 2 ];
		w[ idx ] = w[ idx - ip / 2 ];
	}

	idx += ip * 3 / 4;

	if( id < idLast )
	{
		z[ idx ] = z[ idx + ip / 2 ];
		w[ idx ] = w[ idx + ip / 2 ];
	}
}

}
