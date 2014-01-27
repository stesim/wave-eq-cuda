#include "spdiamat.h"

namespace CudaHelper7
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
		double2* a,
		double s,
		const double2* b )
{
	for( unsigned int i = 0; i < ip; ++i )
	{
		a[ i * stride ].x += s * b[ i * stride ].x;
		a[ i * stride ].y += s * b[ i * stride ].y;
	}
}

__device__
inline void addScaledVecs(
		unsigned int stride,
		unsigned int ip,
		double s,
		const double2* a,
		double t,
		const double2* b,
		double2* c )
{
	for( unsigned int i = 0; i < ip; ++i )
	{
		c[ i * stride ].x = s * a[ i * stride ].x + t * b[ i * stride ].x;
		c[ i * stride ].y = s * a[ i * stride ].y + t * b[ i * stride ].y;
	}
}

__global__
void calculateFirstStep(
		double dt,
		double h,
		unsigned int ip,
		SpDiaMat matInner,
		SpDiaMat matLeft,
		SpDiaMat matRight,
		const double2* F,
		const double2* G,
		double2* Z )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 1;
	unsigned int vecIdx = blockIdx.x * blockDim.x * ip + threadIdx.x;
	const double2* f = &F[ vecIdx ];
	const double2* g = &G[ vecIdx ];
	double2* z = &Z[ vecIdx ];

	SpDiaMat* mat;
	SpDiaMat* mat2;
	if( id == 0 )
	{
		mat = &matLeft;
		mat2 = &matInner;
	}
	else if( id == idLast )
	{
		mat = &matRight;
		mat2 = &matRight;
	}
	else
	{
		mat = &matInner;
		mat2 = &matInner;
	}

	mulMatVec( 2 * blockDim.x, mat, (const double*)f, (double*)z );
	mulMatVec( 2 * blockDim.x, mat2, (const double*)f + 1, (double*)z + 1 );
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
		double2* Z,
		double2* W,
		double2* U )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 1;
	unsigned int vecIdx = blockIdx.x * blockDim.x * ip + threadIdx.x;

	double2* z = &Z[ vecIdx ];
	double2* w = &W[ vecIdx ];
	double2* u = &U[ vecIdx ];

	SpDiaMat* mat;
	SpDiaMat* mat2;
	if( id == 0 )
	{
		mat = &matLeft;
		mat2 = &matInner;
	}
	else if( id == idLast )
	{
		mat = &matRight;
		mat2 = &matRight;
	}
	else
	{
		mat = &matInner;
		mat2 = &matInner;
	}

	double2* swap;
	for( unsigned int i = 0; i < nsteps; ++i )
	{
		mulMatVec( blockDim.x, mat, (const double*)z, (double*)u );
		mulMatVec( blockDim.x, mat2, (double*)z + 1, (double*)u + 1 );
		addVecScaledVec( blockDim.x, ip, u, a, z );
		addVecScaledVec( blockDim.x, ip, u, -1.0, w );

		swap = w;
		w = z;
		z = u;
		u = swap;
	}
}

__device__
inline void memcpy_strided( unsigned int stride, double* dst, const double* src, unsigned int size )
{
	for( unsigned int i = 0; i < size; ++i )
	{
		dst[ i * stride ] = src[ i * stride ];
	}
}

__global__
void synchronizeResults( unsigned int ip, double2* Z, double2* W )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 1;
	unsigned int vecIdx = blockIdx.x * blockDim.x * ip + threadIdx.x;
	double2* z = &Z[ vecIdx ];
	double2* w = &W[ vecIdx ];
	
	//const unsigned int copySize = ip / 4 * sizeof( double );

	unsigned int stride = 2 * blockDim.x;
	/*
	double* nz;
	double* nw;
	*/
	if( id > 0 )
	{
		int offset = threadIdx.x > 0 ? -1 : -(int)stride * ( ip - 1 ) - 1;
		/*
		nz = (double*)z + offset;
		nw = (double*)w + offset;
		*/
		memcpy_strided( stride, (double*)z, (double*)z + offset + ip / 2 * stride, ip / 4 );
		memcpy_strided( stride, (double*)w, (double*)w + offset + ip / 2 * stride, ip / 4 );
		memcpy_strided( stride, (double*)z + 1, (double*)z + ip / 2 * stride, ip / 4 );
		memcpy_strided( stride, (double*)w + 1, (double*)w + ip / 2 * stride, ip / 4 );
		//memcpy( &z[ 0 ], &nz[ ip / 2 ], copySize );
		//memcpy( &w[ 0 ], &nw[ ip / 2 ], copySize );
	}
	else
	{
		/*
		int offset = 0;
		nz = z + offset;
		nw = w + offset;
		*/
		memcpy_strided( stride, (double*)z + 1, (double*)z + ip / 2 * stride, ip / 4 );
		memcpy_strided( stride, (double*)w + 1, (double*)w + ip / 2 * stride, ip / 4 );
		//memcpy( &z[ 0 ], &nz[ ip / 2 ], copySize );
		//memcpy( &w[ 0 ], &nw[ ip / 2 ], copySize );
	}
	if( id < idLast )
	{
		int offset = threadIdx.x < blockDim.x - 1 ? 1 : stride * ( ip - 1 ) + 1;
		/*
		nz = z + offset;
		nw = w + offset;
		*/
		memcpy_strided( stride, (double*)z + 1 + ip * 3 / 4 * stride, (double*)z + offset + ip / 4 * stride, ip / 4 );
		memcpy_strided( stride, (double*)w + 1 + ip * 3 / 4 * stride, (double*)w + offset + ip / 4 * stride, ip / 4 );
		memcpy_strided( stride, (double*)z + ip * 3 / 4 * stride, (double*)z + 1 + ip / 4 * stride, ip / 4 );
		memcpy_strided( stride, (double*)w + ip * 3 / 4 * stride, (double*)w + 1 + ip / 4 * stride, ip / 4 );
		//memcpy( &z[ ip * 3 / 4 ], &nz[ ip / 4 ], copySize );
		//memcpy( &w[ ip * 3 / 4 ], &nw[ ip / 4 ], copySize );
	}
}

}
