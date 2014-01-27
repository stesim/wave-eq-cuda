#include "spdiamat.h"

namespace CudaHelper3
{

__device__
inline void mulMatVec( const SpDiaMat* mat, const double* v, double* res )
{
	for( unsigned int i = 0; i < mat->n; ++i )
	{
		res[ i ] = 0.0;
	}
	const double* curDiagValues = mat->values;
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
		double* a,
		double s,
		const double* b )
{
	for( unsigned int i = 0; i < ip; ++i )
	{
		a[ i ] += s * b[ i ];
	}
}

__device__
inline void addScaledVecs(
		unsigned int ip,
		double s,
		const double* a,
		double t,
		const double* b,
		double* c )
{
	for( unsigned int i = 0; i < ip; ++i )
	{
		c[ i ] = s * a[ i ] + t * b[ i ];
	}
}

extern __shared__ double sharedBuffer[];

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
	
	double* Fs = sharedBuffer;
	double* Gs = sharedBuffer + blockDim.x * ip;
	double* Zs = sharedBuffer + 2 * blockDim.x * ip;

	double* f = &Fs[ threadIdx.x * ip ];
	double* g = &Gs[ threadIdx.x * ip ];
	double* z = &Zs[ threadIdx.x * ip ];

	memcpy( f, &F[ id * ip ], ip * sizeof( double ) );
	memcpy( g, &G[ id * ip ], ip * sizeof( double ) );
	memcpy( z, &Z[ id * ip ], ip * sizeof( double ) );

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

	memcpy( &Z[ id * ip ], z, ip * sizeof( double ) );
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
	
	double* Zs = sharedBuffer;
	double* Ws = sharedBuffer + blockDim.x * ip;
	double* Us = sharedBuffer + 2 * blockDim.x * ip;

	double* z = &Zs[ threadIdx.x * ip ];
	double* w = &Ws[ threadIdx.x * ip ];
	double* u = &Us[ threadIdx.x * ip ];

	memcpy( z, &Z[ id * ip ], ip * sizeof( double ) );
	memcpy( w, &W[ id * ip ], ip * sizeof( double ) );
	memcpy( u, &U[ id * ip ], ip * sizeof( double ) );

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

	double* swap;
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

	memcpy( &Z[ id * ip ], z, ip * sizeof( double ) );
	memcpy( &W[ id * ip ], w, ip * sizeof( double ) );
	memcpy( &U[ id * ip ], u, ip * sizeof( double ) );
}

__global__
void synchronizeResults( unsigned int ip, double* Z, double* W )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 2;
	double* z = &Z[ id * ip ];
	double* w = &W[ id * ip ];
	
	const unsigned int copySize = ip / 4 * sizeof( double );

	double* nz;
	double* nw;
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
