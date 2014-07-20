#include "spdiamat.h"

namespace CudaHelper7
{

// sparse matrix-vector multiplication function for the SpDiaMat structure
__device__
inline void mulMatVec( unsigned int stride, const SpDiaMat* mat,
		const double2* v, double2* res )
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

// vector copy helper function
__device__
inline void memcpy_vec( unsigned int stride, double2* dst, const double2* src,
		unsigned int size )
{
	for( unsigned int i = 0; i < size; ++i )
	{
		dst[ i * stride ].x = src[ i * stride ].x;
		dst[ i * stride ].y = src[ i * stride ].y;
	}
}

// calculate the first time step of the simulation
__global__
void calculateFirstStep( double dt, double h, double a, double b,
		unsigned int ip, SpDiaMat matInner, SpDiaMat matLeft, SpDiaMat matRight,
		const double2* F, const double2* G, double2* Z )
{
	// calculate current subdomain's position in memory
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idLast = gridDim.x * blockDim.x - 2;
	unsigned int vecIdx = blockIdx.x * blockDim.x * ip / 2 + threadIdx.x;

	const double2* f = &F[ vecIdx ];
	const double2* g = &G[ vecIdx ];
	double2* z = &Z[ vecIdx ];

	// select right matrix depending on the subdomain's position
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

	// calculate first step
	mulMatVec( blockDim.x, mat, f, z );
	for( unsigned int i = 0; i < ip / 2; ++i )
	{
		unsigned int idx = i * blockDim.x;
		double f3 = f[ idx ].x;
		f3 *= f3 * f3;
		z[ idx ].x = 0.5 * z[ idx ].x + a * f[ idx ].x + dt * g[ idx ].x + b * f3;
		f3 = f[ idx ].y;
		f3 *= f3 * f3;
		z[ idx ].y = 0.5 * z[ idx ].y + a * f[ idx ].y + dt * g[ idx ].y + b * f3;
	}
}

// calculate a full cycle before rejoining the solutions - basically the same as
// calculating the first step except for a loop over the number of steps
__global__
void calculateNSteps( unsigned int nsteps, double a, double b, unsigned int ip,
		SpDiaMat matInner, SpDiaMat matLeft, SpDiaMat matRight,
		double2* Z, double2* W, double2* U )
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

	double2* swap;
	for( unsigned int i = 0; i < nsteps; ++i )
	{
		mulMatVec( blockDim.x, mat, z, u );
		for( unsigned int i = 0; i < ip / 2; ++i )
		{
			unsigned int idx = i * blockDim.x;
			double z3 = z[ idx ].x;
			z3 *= z3 * z3;
			u[ idx ].x += a * z[ idx ].x - w[ idx ].x + b * z3;
			z3 = z[ idx ].y;
			z3 *= z3 * z3;
			u[ idx ].y += a * z[ idx ].y - w[ idx ].y + b * z3;
		}

		// swap vector pointers to avoid copy operations
		swap = w;
		w = z;
		z = u;
		u = swap;
	}
}

// replace invalid values with the correct ones by copying them from the
// neighboring subdomains
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
