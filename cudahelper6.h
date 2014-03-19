/* Coalesced scalar */

#include "spdiamat.h"

namespace CudaHelper6
{

__device__
inline void mulMatVec( unsigned int stride, const SpDiaMat* mat, const double* v, double* res )
{
	for( unsigned int i = 0; i < mat->n; ++i )
	{
		res[ i * stride ] = 0.0;
	}
	const double* curDiagValues = mat->values;
	for( unsigned int k = 0; k < mat->diags; ++k )
	{
		int offset = mat->offsets[ k ];
		int resIdx = -min( offset, 0 );
		int vIdx = max( offset, 0 );
		unsigned int diagSize = mat->n - abs( offset );
		for( unsigned int i = 0; i < diagSize; ++i )
		{
			res[ ( i + resIdx ) * stride ] +=
				curDiagValues[ i ] * v[ ( i + vIdx ) * stride ];
		}
		curDiagValues += diagSize;
	}
}

__device__
inline void addVecScaledVec(
		unsigned int stride,
		unsigned int ip,
		double* a,
		double s,
		const double* b )
{
	for( unsigned int i = 0; i < ip; ++i )
	{
		a[ i * stride ] += s * b[ i * stride ];
	}
}

__device__
inline void addScaledVecs(
		unsigned int stride,
		unsigned int ip,
		double s,
		const double* a,
		double t,
		const double* b,
		double* c )
{
	for( unsigned int i = 0; i < ip; ++i )
	{
		c[ i * stride ] = s * a[ i * stride ] + t * b[ i * stride ];
	}
}

/*
__global__
void copyInitialValuesCoalesced(
		unsigned int ip,
		const double* f,
		const double* g,
		double* w,
		double* u )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int srcIdx = id * ip / 2;
	unsigned int coalescedIdx = blockIdx.x * blockDim.x * ip + threadIdx.x;

	unsigned int stride = blockDim.x;
	if( id < gridDim.x * blockDim.x - 1 )
	{
		for( unsigned int i = 0; i < ip; ++i )
		{
			w[ coalescedIdx + i * stride ] = f[ srcIdx + i ];
			u[ coalescedIdx + i * stride ] = g[ srcIdx + i ];
		}
	}
}

__global__
void copyResultsLinear(
		unsigned int ip,
		const double* z,
		double* solution )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int srcIdx = id * ip / 2;
	unsigned int coalescedIdx = blockIdx.x * blockDim.x * ip + threadIdx.x;

	unsigned int stride = blockDim.x;
	if( id < gridDim.x * blockDim.x - 1 )
	{
		for( unsigned int i = 0; i < ip / 2; ++i )
		{
			solution[ srcIdx + i ] = z[ coalescedIdx + i * stride ];
		}
	}
}
*/

__global__
void calculateFirstStep(
		double dt,
		double h,
		unsigned int ip,
		SpDiaMat matInner,
		SpDiaMat matLeft,
		SpDiaMat matRight,
		const double* F,
		const double* G,
		double* Z )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 2;
	unsigned int vecIdx = blockIdx.x * blockDim.x * ip + threadIdx.x;
	const double* f = &F[ vecIdx ];
	const double* g = &G[ vecIdx ];
	double* z = &Z[ vecIdx ];

	SpDiaMat* mat;
	if( id == 0 )
	{
		mat = &matLeft;
	}
	else if( id == idLast )
	{
		mat = &matRight;
	}
	else
	{
		mat = &matInner;
	}

	mulMatVec( blockDim.x, mat, f, z );
	addScaledVecs( blockDim.x, ip, 0.5, z, ( 1.0 - dt / h * dt / h ), f, z );
	addVecScaledVec( blockDim.x, ip, z, dt, g );
}

__global__
void calculateNSteps(
		unsigned int nsteps,
		double a,
		unsigned int ip,
		SpDiaMat matInner,
		SpDiaMat matLeft,
		SpDiaMat matRight,
		double* Z,
		double* W,
		double* U )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 2;
	unsigned int vecIdx = blockIdx.x * blockDim.x * ip + threadIdx.x;

	SpDiaMat* mat;
	if( id == 0 )
	{
		mat = &matLeft;
	}
	else if( id == idLast )
	{
		mat = &matRight;
	}
	else
	{
		mat = &matInner;
	}

	double* z = &Z[ vecIdx ];
	double* w = &W[ vecIdx ];
	double* u = &U[ vecIdx ];

	double* swap;
	for( unsigned int i = 0; i < nsteps; ++i )
	{
		mulMatVec( blockDim.x, mat, z, u );
		addVecScaledVec( blockDim.x, ip, u, a, z );
		addVecScaledVec( blockDim.x, ip, u, -1.0, w );

		swap = w;
		w = z;
		z = u;
		u = swap;
	}
}

__device__
inline void memcpy_vec( unsigned int stride, double* dst, const double* src, unsigned int size )
{
	for( unsigned int i = 0; i < size; ++i )
	{
		dst[ i * stride ] = src[ i * stride ];
	}
}

__global__
void synchronizeResults( unsigned int ip, double* Z, double* W )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 2;
	unsigned int vecIdx = blockIdx.x * blockDim.x * ip + threadIdx.x;
	double* z = &Z[ vecIdx ];
	double* w = &W[ vecIdx ];
	
	//const unsigned int copySize = ip / 4 * sizeof( double );

	unsigned int stride = blockDim.x;
	double* nz;
	double* nw;
	if( id > 0 )
	{
		int offset = threadIdx.x > 0 ? -1 : -(int)stride * ( ip - 1 ) - 1;
		nz = z + offset;
		nw = w + offset;
		memcpy_vec( stride, &z[ 0 ], &nz[ ip / 2 * stride ], ip / 4 );
		memcpy_vec( stride, &w[ 0 ], &nw[ ip / 2 * stride ], ip / 4 );
		//memcpy( &z[ 0 ], &nz[ ip / 2 ], copySize );
		//memcpy( &w[ 0 ], &nw[ ip / 2 ], copySize );
	}
	if( id < idLast )
	{
		int offset = threadIdx.x < blockDim.x - 1 ? 1 : stride * ( ip - 1 ) + 1;
		nz = z + offset;
		nw = w + offset;
		memcpy_vec( stride, &z[ ip * 3 / 4 * stride ], &nz[ ip / 4 * stride ], ip / 4 );
		memcpy_vec( stride, &w[ ip * 3 / 4 * stride ], &nw[ ip / 4 * stride ], ip / 4 );
		//memcpy( &z[ ip * 3 / 4 ], &nz[ ip / 4 ], copySize );
		//memcpy( &w[ ip * 3 / 4 ], &nw[ ip / 4 ], copySize );
	}
}

}
