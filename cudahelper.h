/* Work on entire domain */

#pragma once

#include "spdiamat.h"

namespace CudaHelper
{

__device__
inline double diaMulVec( unsigned int row, const SpDiaMat* mat, const double* a )
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
		int* offsets,
		double* diaValues,
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

	z[ row ] = 0.5 * diaMulVec( row, &mat, f ) + ( 1.0 - dt / h * dt / h )
		* f[ row ] + dt * g[ row ];
}

__global__
void calculateNextStep(
		unsigned int ip,
		double a,
		unsigned int diags,
		int* offsets,
		double* diaValues,
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

