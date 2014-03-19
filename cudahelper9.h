/* Coalesced vectorized + constant matrices */

#include "spdiamat.h"

namespace CudaHelper9
{

static const unsigned int CONST_BUFFER_SIZE = 32768;
//static const unsigned int CONST_BUFFER_SIZE = 8192;
__constant__ char constBuffer[ CONST_BUFFER_SIZE ];

__device__
inline void mulMatVec( unsigned int stride, const SpDiaMat* mat, const double2* v, double2* res )
{
	for( unsigned int i = 0; i < mat->n / 2; ++i )
	{
		res[ i * stride ].x = 0.0;
		res[ i * stride ].y = 0.0;
	}
	stride *= 2;
	const double* _v = (const double*)v;
	double* _res = (double*)res;
	const double* curDiagValues = mat->values;
	for( unsigned int k = 0; k < mat->diags; ++k )
	{
		int offset = mat->offsets[ k ];
		int resIdx = -min( offset, 0 );
		int vIdx = max( offset, 0 );
		unsigned int diagSize = mat->n - abs( offset );
		for( unsigned int i = 0; i < diagSize; ++i )
		{
			_res[ ( ( i + resIdx ) / 2 ) * stride + ( ( i + resIdx ) & 1 ) ] +=
				curDiagValues[ i ] * _v[ ( ( i + vIdx ) / 2 ) * stride + ( ( i + vIdx ) & 1 ) ];
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
	for( unsigned int i = 0; i < ip / 2; ++i )
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
	for( unsigned int i = 0; i < ip / 2; ++i )
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
	unsigned int idLast = gridDim.x * blockDim.x - 2;
	unsigned int vecIdx = blockIdx.x * blockDim.x * ip / 2 + threadIdx.x;
	const double2* f = &F[ vecIdx ];
	const double2* g = &G[ vecIdx ];
	double2* z = &Z[ vecIdx ];

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
	mat->offsets = (int*)( constBuffer + mat->constPos );
	mat->values = (double*)( mat->offsets + mat->diags );

	mulMatVec( blockDim.x, mat, f, z );
	addScaledVecs( blockDim.x, ip, 0.5, z, ( 1.0 - dt / h * dt / h ), f, z );
	addScaledVecs( blockDim.x, ip, 1.0, z, dt, g, z );
	//addVecScaledVec( blockDim.x, ip, z, dt, g );
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
	unsigned int idLast = gridDim.x * blockDim.x - 2;
	unsigned int vecIdx = blockIdx.x * blockDim.x * ip / 2 + threadIdx.x;

	double2* z = &Z[ vecIdx ];
	double2* w = &W[ vecIdx ];
	double2* u = &U[ vecIdx ];

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
	mat->offsets = (int*)( constBuffer + mat->constPos );
	mat->values = (double*)( mat->offsets + mat->diags );

	double2* swap;
	for( unsigned int i = 0; i < nsteps; ++i )
	{
		mulMatVec( blockDim.x, mat, z, u );
		addScaledVecs( blockDim.x, ip, 1.0, u, a, z, u );
		addScaledVecs( blockDim.x, ip, 1.0, u, -1.0, w, u );
		//addVecScaledVec( blockDim.x, ip, u, a, z );
		//addVecScaledVec( blockDim.x, ip, u, -1.0, w );

		swap = w;
		w = z;
		z = u;
		u = swap;
	}
}

__device__
inline void memcpy_vec( unsigned int stride, double2* dst, const double2* src, unsigned int size )
{
	for( unsigned int i = 0; i < size; ++i )
	{
		dst[ i * stride ].x = src[ i * stride ].x;
		dst[ i * stride ].y = src[ i * stride ].y;
	}
}

__global__
void synchronizeResults( unsigned int ip, double2* Z, double2* W )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 2;
	unsigned int vecIdx = blockIdx.x * blockDim.x * ip / 2 + threadIdx.x;
	double2* z = &Z[ vecIdx ];
	double2* w = &W[ vecIdx ];
	
	const unsigned int copySize = ip / 8;
	unsigned int stride = blockDim.x;
	if( id > 0 )
	{
		int offset = -( threadIdx.x > 0 ? 1 : stride * ( ip / 2 - 1 ) + 1 );
		memcpy_vec( stride, z, z + offset + ip / 4 * stride, copySize );
		memcpy_vec( stride, w, w + offset + ip / 4 * stride, copySize );
	}
	if( id < idLast )
	{
		int offset = threadIdx.x < blockDim.x - 1 ? 1 : stride * ( ip / 2 - 1 ) + 1;
		memcpy_vec( stride, z + ip / 2 * 3 / 4 * stride, z + offset + ip / 8 * stride, copySize );
		memcpy_vec( stride, w + ip / 2 * 3 / 4 * stride, w + offset + ip / 8 * stride, copySize );
	}
}

}
