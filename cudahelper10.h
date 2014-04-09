/* Coalesced vectorized + hardcoded matrix multiplication */

#include "spdiamat.h"

namespace CudaHelper10
{

__device__
inline void mulMatVec( unsigned int stride, unsigned int ip, real l2, unsigned int type, const real2* v, real2* res )
{
	unsigned int pos0 = 0;
	unsigned int pos1 = stride;

	res[ pos0 ].x = ( type == 0 ? 2.0 * l2 : l2 ) * v[ pos0 ].y;
	for( unsigned int i = 0; i < ip - 1; ++i )
	{
		res[ pos0 ].y = l2 * ( v[ pos0 ].x + v[ pos1 ].x );
		res[ pos1 ].x = l2 * ( v[ pos0 ].y + v[ pos1 ].y );

		pos0 = pos1;
		pos1 += stride;
	}
	res[ pos0 ].y = ( type == 2 ? 2.0 * l2 : l2 ) * v[ pos0 ].x;
}

__global__
void calculateFirstStep(
		real dt,
		real h,
		unsigned int ip,
		const real2* f,
		const real2* g,
		real2* z )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int vecIdx = blockIdx.x * blockDim.x * ip + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 2;
	f += vecIdx;
	g += vecIdx;
	z += vecIdx;

	real l2 = dt / h * dt / h;
	real a = 1.0 - l2;

	unsigned int type;
	if( id == 0 )
	{
		type = 0;
	}
	else if( id == idLast )
	{
		type = 2;
	}
	else
	{
		type = 1;
	}

	mulMatVec( blockDim.x, ip, l2, type, f, z );
	for( unsigned int j = 0; j < ip; ++j )
	{
		unsigned int idx = j * blockDim.x;
		z[ idx ].x = 0.5 * z[ idx ].x + a * f[ idx ].x + dt * g[ idx ].x;
		z[ idx ].y = 0.5 * z[ idx ].y + a * f[ idx ].y + dt * g[ idx ].y;
	}
}

__global__
void calculateNSteps(
		unsigned int nsteps,
		real l2,
		real a,
		unsigned int ip,
		real2* z,
		real2* w,
		real2* u )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int vecIdx = blockIdx.x * blockDim.x * ip + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 2;

	z += vecIdx;
	w += vecIdx;
	u += vecIdx;

	unsigned int type;
	if( id == 0 )
	{
		type = 0;
	}
	else if( id == idLast )
	{
		type = 2;
	}
	else
	{
		type = 1;
	}

	real2* swap;
	for( unsigned int i = 0; i < nsteps; ++i )
	{
		mulMatVec( blockDim.x, ip, l2, type, z, u );
		for( unsigned int j = 0; j < ip; ++j )
		{
			unsigned int idx = j * blockDim.x;
			u[ idx ].x += a * z[ idx ].x - w[ idx ].x;
			u[ idx ].y += a * z[ idx ].y - w[ idx ].y;
		}

		swap = w;
		w = z;
		z = u;
		u = swap;
	}
}

__device__
inline void memcpy_vec( unsigned int stride, real2* dst, const real2* src, unsigned int size )
{
	for( unsigned int i = 0; i < size; ++i )
	{
		dst[ i * stride ].x = src[ i * stride ].x;
		dst[ i * stride ].y = src[ i * stride ].y;
	}
}

__global__
void synchronizeResults( unsigned int ip, real2* z, real2* w )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 2;
	unsigned int vecIdx = blockIdx.x * blockDim.x * ip + threadIdx.x;
	z += vecIdx;
	w += vecIdx;
	
	const unsigned int copySize = ip / 4;
	unsigned int stride = blockDim.x;
	if( id > 0 )
	{
		int offset = -( threadIdx.x > 0 ? 1 : stride * ( ip - 1 ) + 1 );
		memcpy_vec( stride, z, z + offset + ip / 2 * stride, copySize );
		memcpy_vec( stride, w, w + offset + ip / 2 * stride, copySize );
	}
	if( id < idLast )
	{
		int offset = threadIdx.x < blockDim.x - 1 ? 1 : stride * ( ip - 1 ) + 1;
		memcpy_vec( stride, z + ip * 3 / 4 * stride, z + offset + ip / 4 * stride, copySize );
		memcpy_vec( stride, w + ip * 3 / 4 * stride, w + offset + ip / 4 * stride, copySize );
	}
}

}
