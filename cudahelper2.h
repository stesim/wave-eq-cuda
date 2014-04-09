/* Standard */

#include "spdiamat.h"

namespace CudaHelper2
{

__device__
inline void mulMatVec( const SpDiaMat* mat, const real* v, real* res )
{
	for( unsigned int i = 0; i < mat->n; ++i )
	{
		res[ i ] = 0.0;
	}
	real* curDiagValues = mat->values;
	for( unsigned int k = 0; k < mat->diags; ++k )
	{
		int offset = mat->offsets[ k ];
		unsigned int diagSize = mat->n - abs( offset );
		for( unsigned int i = 0; i < diagSize; ++i )
		{
			res[ i - min( offset, 0 ) ] +=
				curDiagValues[ i ] * v[ i + max( offset, 0 ) ];
		}
		curDiagValues += diagSize;
	}
}

__device__
inline void addVecScaledVec(
		unsigned int ip,
		real* a,
		real s,
		const real* b )
{
	for( unsigned int i = 0; i < ip; ++i )
	{
		a[ i ] += s * b[ i ];
	}
}

__device__
inline void addScaledVecs(
		unsigned int ip,
		real s,
		const real* a,
		real t,
		const real* b,
		real* c )
{
	for( unsigned int i = 0; i < ip; ++i )
	{
		c[ i ] = s * a[ i ] + t * b[ i ];
	}
}

__global__
void calculateFirstStep(
		real dt,
		real h,
		unsigned int ip,
		SpDiaMat matInner,
		SpDiaMat matLeft,
		SpDiaMat matRight,
		const real* F,
		const real* G,
		real* Z )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 2;
	const real* f = &F[ id * ip ];
	const real* g = &G[ id * ip ];
	real* z = &Z[ id * ip ];

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

	mulMatVec( mat, f, z );
	addScaledVecs( ip, 0.5, z, ( 1.0 - dt / h * dt / h ), f, z );
	addVecScaledVec( ip, z, dt, g );
}

__global__
void calculateNSteps(
		unsigned int nsteps,
		real a,
		unsigned int ip,
		SpDiaMat matInner,
		SpDiaMat matLeft,
		SpDiaMat matRight,
		real* Z,
		real* W,
		real* U )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 2;

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

	real* z = &Z[ id * ip ];
	real* w = &W[ id * ip ];
	real* u = &U[ id * ip ];

	real* swap;
	for( unsigned int i = 0; i < nsteps; ++i )
	{
		mulMatVec( mat, z, u );
		addVecScaledVec( ip, u, a, z );
		addVecScaledVec( ip, u, -1.0, w );

		swap = w;
		w = z;
		z = u;
		u = swap;
	}
}

__global__
void synchronizeResults( unsigned int ip, real* Z, real* W )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 2;
	real* z = &Z[ id * ip ];
	real* w = &W[ id * ip ];
	
	const unsigned int copySize = ip / 4 * sizeof( real );

	real* nz;
	real* nw;
	if( id > 0 )
	{
		nz = &Z[ ( id - 1 ) * ip ];
		nw = &W[ ( id - 1 ) * ip ];
		memcpy( &z[ 0 ], &nz[ ip / 2 ], copySize );
		memcpy( &w[ 0 ], &nw[ ip / 2 ], copySize );
	}
	if( id < idLast )
	{
		nz = &Z[ ( id + 1 ) * ip ];
		nw = &W[ ( id + 1 ) * ip ];
		memcpy( &z[ ip * 3 / 4 ], &nz[ ip / 4 ], copySize );
		memcpy( &w[ ip * 3 / 4 ], &nw[ ip / 4 ], copySize );
	}
}

}
