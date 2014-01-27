#include "spdiamat.h"

namespace CudaHelper4
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

	const double* f = &F[ id * ip ];
	const double* g = &G[ id * ip ];
	double* z = &Z[ id * ip ];

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

	double* z = &Z[ id * ip ];
	double* w = &W[ id * ip ];
	double* u = &U[ id * ip ];

	SpDiaMat mat;
	if( id == 0 )
	{
		mat.n = matLeft.n;
		mat.diags = matLeft.diags;
		mat.valuesSize = matLeft.valuesSize;
		mat.values = (double*)( (char*)sharedBuffer + matInner.valuesSize * sizeof( double ) + matInner.diags * sizeof( int ) );
		mat.offsets = (int*)( &mat.values[ mat.valuesSize ] );

		memcpy( mat.values, matLeft.values, mat.valuesSize * sizeof( double ) );
		memcpy( mat.offsets, matLeft.offsets, mat.diags * sizeof( int ) );
	}
	else if( id == idLast )
	{
		mat.n = matRight.n;
		mat.diags = matRight.diags;
		mat.valuesSize = matRight.valuesSize;
		mat.values = (double*)( (char*)sharedBuffer + ( matInner.valuesSize + matLeft.valuesSize ) * sizeof( double ) + ( matInner.diags + matLeft.diags ) * sizeof( int ) );
		mat.offsets = (int*)( &mat.values[ mat.valuesSize ] );
		
		memcpy( mat.values, matRight.values, mat.valuesSize * sizeof( double ) );
		memcpy( mat.offsets, matRight.offsets, mat.diags * sizeof( int ) );
	}
	else
	{
		mat.n = matInner.n;
		mat.diags = matInner.diags;
		mat.valuesSize = matInner.valuesSize;
		mat.values = sharedBuffer;
		mat.offsets = (int*)( &mat.values[ mat.valuesSize ] );

		if( threadIdx.x == 1 )
		{
			memcpy( mat.values, matInner.values, mat.valuesSize * sizeof( double ) );
			memcpy( mat.offsets, matInner.offsets, mat.diags * sizeof( int ) );
		}
	}

	__syncthreads();

	double* swap;
	for( unsigned int i = 0; i < nsteps; ++i )
	{
		mulMatVec( &mat, z, u );
		addVecScaledVec( ip, u, a, z );
		addVecScaledVec( ip, u, -1.0, w );

		swap = w;
		w = z;
		z = u;
		u = swap;
	}
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
