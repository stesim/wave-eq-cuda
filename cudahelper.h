#pragma once

/*
#ifndef __global__
#define __global__
#define __device__
#define abs(x) ((x<0)?-x:x)
#define max(a,b) ((a>b)?a:b)
struct _dim { unsigned int x; };
extern _dim blockIdx;
extern _dim blockDim;
extern _dim threadIdx;
#endif
*/

namespace CudaHelper
{

struct SpDiaMat
{
	unsigned int n;
	unsigned int diags;
	const int* offsets;
	const double* values;
};

__device__
double diaMulVec( unsigned int row, const SpDiaMat* mat, const double* a )
{
	double val = 0.0;
	const double* curDiagValues = mat->values;
	for( unsigned int i = 0; i < mat->diags; ++i )
	{
		int offset = mat->offsets[ i ];
		int col = row + offset;
		if( col >= 0 && col < (int)( mat->n ) )
		{
			val += a[ col ] * curDiagValues[ col - max( offset, 0 ) ];
		}
		curDiagValues += mat->n - abs( offset );
	}
	return val;
}

__global__
void calculateFirstStep(
		double dt,
		double h,
		unsigned int ip,
		unsigned int diags,
		const int* offsets,
		const double* diaValues,
		const double* f,
		const double* g,
		double* z )
{
	unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = diags;
	mat.offsets = offsets;
	mat.values = diaValues;

	// z = M * w
	//val += diaMulVec( row, &mat, f );
	// z = 0.5 * z + (1 - (dt / h)^2) * w = 0.5 * M * w + (1 - (dt / h)^2) * w
	//vecAddScaledVecs( ip, 0.5, z, ( 1.0 - dt / h * dt / h ), f, z );
	// z = z + dt * u
	//vecAddScaledVec( ip, z, dt, g );

	z[ row ] = 0.5 * diaMulVec( row, &mat, f ) + ( 1.0 - dt / h * dt / h )
		* f[ row ] + dt * g[ row ];
}

__global__
void calculateNextStep(
		unsigned int ip,
		double a,
		unsigned int diags,
		const int* offsets,
		const double* diaValues,
		double* z,
		double* w,
		double* u )
{
	unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = diags;
	mat.offsets = offsets;
	mat.values = diaValues;

	u[ row ] = diaMulVec( row, &mat, z ) + a * z[ row ] - w[ row ];
}

}

