#include "solver9.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include "cudahelper9.h"
#include <iostream>

Solver9::Solver9()
{
}

Solver9::~Solver9()
{
}

void Solver9::solve()
{
	cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

	unsigned int ip = np / ns;
	double h = 2 * L / ( np - 1 );
	double dt = h / 4.0;
	double l = dt / h;
	double l2 = l * l;
	unsigned int nsteps = ip / 2;
	unsigned int ndom = 2 * ns - 1;
	unsigned int gpudom = 2 * ns;
	unsigned int blocks = gpudom / threads;

	unsigned int kmax = ceil( T / ( nsteps * dt ) );
	solution = std::vector<double>( np );
	error = std::vector<double>( kmax );

	std::vector<double> z( gpudom * ip );
	std::vector<double> w( gpudom * ip );
	std::vector<double> u( gpudom * ip );

	unsigned int bufSize = gpudom * ip * sizeof( double );

	double2* d_z;
	double2* d_w;
	double2* d_u;
	cudaMalloc( &d_z, bufSize );
	cudaMalloc( &d_w, bufSize );
	cudaMalloc( &d_u, bufSize );

	std::vector<double> x( np );
	std::vector<double> f( np );
	std::vector<double> g( np );
	for( unsigned int i = 0; i < np; ++i )
	{
		x[ i ] = -L + i * h;
		f[ i ] = u0( x[ i ] );
		g[ i ] = u1( x[ i ] );
	}
	for( unsigned int n = 0; n < blocks; ++n )
	{
		for( unsigned int j = 0; j < threads; ++j )
		{
			if( n == blocks - 1 && j == threads - 1 )
			{
				continue;
			}

			unsigned int base = n * threads * ip + 2 * j;
			unsigned int srcBase = ( n * threads + j ) * ip / 2;
			for( unsigned int i = 0; i < ip / 2; ++i )
			{
				w[ base + 2 * i * threads ] = f[ srcBase + 2 * i ];
				w[ base + 2 * i * threads + 1 ] = f[ srcBase + 2 * i + 1 ];
				u[ base + 2 * i * threads ] = g[ srcBase + 2 * i ];
				u[ base + 2 * i * threads + 1 ] = g[ srcBase + 2 * i + 1 ];
			}
		}
	}

	cudaMemcpy( d_w, w.data(), bufSize, cudaMemcpyHostToDevice );
	cudaMemcpy( d_u, u.data(), bufSize, cudaMemcpyHostToDevice );

	f.clear();
	g.clear();

	unsigned int constBufferOffset = 0;
	SpDiaMat matInner = gpuAllocFDMatrixInner( l2, ip, constBufferOffset );
	constBufferOffset += matInner.diags * sizeof( int ) + matInner.valuesSize * sizeof( double );
	SpDiaMat matLeft = gpuAllocFDMatrixLeft( l2, ip, constBufferOffset );
	constBufferOffset += matLeft.diags * sizeof( int ) + matLeft.valuesSize * sizeof( double );
	SpDiaMat matRight = gpuAllocFDMatrixRight( l2, ip, constBufferOffset );


	/*
	double tmp[ 100 ];
	for( unsigned int i = 0; i < 100; ++i )
	{
		tmp[ i ] = i;
	}
	cudaMemcpyFromSymbol( tmp, CudaHelper9::constBuffer, 100 * sizeof( double ) );
	for( unsigned int i = 0; i < 100; ++i )
	{
		std::cout << tmp[ i ] << std::endl;
	}
	*/


	CudaHelper9::calculateFirstStep<<<blocks, threads>>>(
			dt,	h, ip, matInner, matLeft, matRight,	d_w, d_u, d_z );

	CudaHelper9::synchronizeResults<<<blocks, threads>>>(
			ip,
			d_z,
			d_w );

		/*
		cudaMemcpy(	z.data(), d_z, bufSize,	cudaMemcpyDeviceToHost );
		for( unsigned int n = 0; n < blocks; ++n )
		{
			for( unsigned int j = 0; j < threads; j += 2 )
			{
				if( n == blocks - 1 && j == threads - 1 )
				{
					continue;
				}

				unsigned int base = n * threads * ip + 2 * j;
				unsigned int dstBase = ( n * threads + j ) * ip / 2;
				for( unsigned int i = 0; i < ip / 2; ++i )
				{
					solution[ dstBase + 2 * i ] = z[ base + 2 * i * threads ];
					solution[ dstBase + 2 * i + 1 ] = z[ base + 2 * i * threads + 1 ];
				}
			}
		}
		for( unsigned int i = 0; i < 32; ++i )
		{
			std::cout << solution[ i ] << std::endl;
		}
		*/

	for( unsigned int k = 0; k < kmax; ++k )
	{
		CudaHelper9::calculateNSteps<<<blocks, threads>>>(
				nsteps,	2.0 * ( 1.0 - l2 ),	ip,	matInner, matLeft, matRight,
				d_z, d_w, d_u );

		double2* swap;
		switch( nsteps % 3 )
		{
			case 1:
				swap = d_w;
				d_w = d_z;
				d_z = d_u;
				d_u = swap;
				break;
			case 2:
				swap = d_z;
				d_z = d_w;
				d_w = d_u;
				d_u = swap;
		}

		CudaHelper9::synchronizeResults<<<blocks, threads>>>(
				ip,
				d_z,
				d_w );

		cudaMemcpy(
				z.data(),
				d_z,
				bufSize,
				cudaMemcpyDeviceToHost );

		for( unsigned int n = 0; n < blocks; ++n )
		{
			for( unsigned int j = 0; j < threads; j += 2 )
			{
				if( n == blocks - 1 && j == threads - 1 )
				{
					continue;
				}

				unsigned int base = n * threads * ip + 2 * j;
				unsigned int dstBase = ( n * threads + j ) * ip / 2;
				for( unsigned int i = 0; i < ip / 2; ++i )
				{
					solution[ dstBase + 2 * i ] = z[ base + 2 * i * threads ];
					solution[ dstBase + 2 * i + 1 ] = z[ base + 2 * i * threads + 1 ];
				}
			}
		}

		double t = ( ( k + 1 ) * nsteps + 1 ) * dt;
		double l2err = 0.0;
		for( unsigned int i = 0; i < np; ++i )
		{
			double err = sol( x[ i ], t ) - solution[ i ];
			l2err += err * err;
		}
		error[ k ] = sqrt( h * l2err );
	}

	solution = z;

	cudaFree( d_z );
	cudaFree( d_w );
	cudaFree( d_u );
	gpuFreeFDMatrix( matInner );
	gpuFreeFDMatrix( matLeft );
	gpuFreeFDMatrix( matRight );
}

SpDiaMat Solver9::gpuAllocFDMatrixInner( double l2, unsigned int ip, unsigned int offset )
{
	unsigned int diagSize = ip - 1;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = 2;
	mat.valuesSize = 2 * diagSize;
	mat.constPos = offset;
	//cudaMalloc( &mat.offsets, 2 * sizeof( int ) );
	//cudaMalloc( &mat.values, 2 * diagSize * sizeof( double ) );

	static const int offsets[] = { 1, -1 };
	double* values = new double[ 2 * diagSize * sizeof( double ) ];

	for( unsigned int i = 0; i < diagSize; ++i )
	{
		values[ i ] = l2;
		values[ diagSize + i ] = l2;
	}

	cudaMemcpyToSymbol( CudaHelper9::constBuffer, offsets, 2 * sizeof( int ), offset, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( CudaHelper9::constBuffer, values, 2 * diagSize * sizeof( double ), offset + 2 * sizeof( int ), cudaMemcpyHostToDevice );
	//cudaMemcpy( const_cast<int*>( mat.offsets ), offsets, 2 * sizeof( int ), cudaMemcpyHostToDevice );
	//cudaMemcpy( const_cast<double*>( mat.values ), values, 2 * diagSize * sizeof( double ), cudaMemcpyHostToDevice );

	delete[] values;
	return mat;
}

SpDiaMat Solver9::gpuAllocFDMatrixLeft( double l2, unsigned int ip, unsigned int offset )
{
	unsigned int diagSize = ip - 1;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = 2;
	mat.valuesSize = 2 * diagSize;
	mat.constPos = offset;
	//cudaMalloc( &mat.offsets, 2 * sizeof( int ) );
	//cudaMalloc( &mat.values, 2 * diagSize * sizeof( double ) );

	static const int offsets[] = { 1, -1 };
	double* values = new double[ 2 * diagSize * sizeof( double ) ];

	for( unsigned int i = 0; i < diagSize; ++i )
	{
		values[ i ] = ( i != 0 ? l2 : 2.0 * l2 );
		values[ diagSize + i ] = l2;
	}

	cudaMemcpyToSymbol( CudaHelper9::constBuffer, offsets, 2 * sizeof( int ), offset, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( CudaHelper9::constBuffer, values, 2 * diagSize * sizeof( double ), offset + 2 * sizeof( int ), cudaMemcpyHostToDevice );
	//cudaMemcpy( const_cast<int*>( mat.offsets ), offsets, 2 * sizeof( int ), cudaMemcpyHostToDevice );
	//cudaMemcpy( const_cast<double*>( mat.values ), values, 2 * diagSize * sizeof( double ), cudaMemcpyHostToDevice );

	delete[] values;
	return mat;
}

SpDiaMat Solver9::gpuAllocFDMatrixRight( double l2, unsigned int ip, unsigned int offset )
{
	unsigned int diagSize = ip - 1;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = 2;
	mat.valuesSize = 2 * diagSize;
	mat.constPos = offset;
	//cudaMalloc( &mat.offsets, 2 * sizeof( int ) );
	//cudaMalloc( &mat.values, 2 * diagSize * sizeof( double ) );

	static const int offsets[] = { 1, -1 };
	double* values = new double[ 2 * diagSize * sizeof( double ) ];

	for( unsigned int i = 0; i < diagSize; ++i )
	{
		values[ i ] = l2;
		values[ diagSize + i ] = ( i != diagSize - 1 ? l2 : 2.0 * l2 );
	}

	cudaMemcpyToSymbol( CudaHelper9::constBuffer, offsets, 2 * sizeof( int ), offset, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( CudaHelper9::constBuffer, values, 2 * diagSize * sizeof( double ), offset + 2 * sizeof( int ), cudaMemcpyHostToDevice );
	//cudaMemcpy( const_cast<int*>( mat.offsets ), offsets, 2 * sizeof( int ), cudaMemcpyHostToDevice );
	//cudaMemcpy( const_cast<double*>( mat.values ), values, 2 * diagSize * sizeof( double ), cudaMemcpyHostToDevice );

	delete[] values;
	return mat;
}

void Solver9::gpuFreeFDMatrix( SpDiaMat& mat )
{
	//cudaFree( mat.offsets );
	mat.offsets = NULL;
	//cudaFree( mat.values );
	mat.values = NULL;
}
