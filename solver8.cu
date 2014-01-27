#include "solver8.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include "cudahelper8.h"
#include <iostream>

Solver8::Solver8()
{
}

Solver8::~Solver8()
{
}

void Solver8::solve()
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

	std::vector<real> z( gpudom * ip );
	std::vector<real> w( gpudom * ip );
	std::vector<real> u( gpudom * ip );

	unsigned int bufSize = gpudom * ip * sizeof( real );

	real* d_z;
	real* d_w;
	real* d_u;
	cudaMalloc( &d_z, bufSize );
	cudaMalloc( &d_w, bufSize );
	cudaMalloc( &d_u, bufSize );

	std::vector<real> x( np );
	std::vector<real> f( np );
	std::vector<real> g( np );
	for( unsigned int i = 0; i < np; ++i )
	{
		x[ i ] = -L + i * h;
		f[ i ] = u0( x[ i ] );
		g[ i ] = u1( x[ i ] );
	}
	/*
	for( unsigned int i = 0; i < ndom; ++i )
	{
		unsigned int pos = i * ip;
		memcpy( &w[ pos ], &f[ pos / 2 ], ip * sizeof( real ) );
		memcpy( &u[ pos ], &g[ pos / 2 ], ip * sizeof( real ) ); 
	}
	*/
	for( unsigned int n = 0; n < blocks; ++n )
	{
		for( unsigned int k = 0; k < threads; ++k )
		{
			if( n == blocks - 1 && k == threads - 1 )
			{
				continue;
			}

			unsigned int base = n * threads * ip + k;
			unsigned int srcBase = ( n * threads + k ) * ip / 2;
			for( unsigned int i = 0; i < ip; ++i )
			{
				w[ base + i * threads ] = f[ srcBase + i ];
				u[ base + i * threads ] = g[ srcBase + i ];
			}
		}
	}
	/*
	real* d_f;
	real* d_g;
	cudaMalloc( &d_f, np * sizeof( real ) );
	cudaMalloc( &d_g, np * sizeof( real ) );
	cudaMemcpy( d_f, f.data(), np * sizeof( real ), cudaMemcpyHostToDevice );
	cudaMemcpy( d_g, g.data(), np * sizeof( real ), cudaMemcpyHostToDevice );
	CudaHelper8::copyInitialValuesCoalesced<<<blocks, threads>>>(
			ip, d_f, d_g, d_w, d_u );
	cudaFree( d_f );
	real* d_solution = d_g;
	*/

	cudaMemcpy( d_w, w.data(), bufSize, cudaMemcpyHostToDevice );
	cudaMemcpy( d_u, u.data(), bufSize, cudaMemcpyHostToDevice );

	f.clear();
	g.clear();

	SpDiaMat matInner = gpuAllocFDMatrixInner( l2, ip );
	SpDiaMat matLeft = gpuAllocFDMatrixLeft( l2, ip );
	SpDiaMat matRight = gpuAllocFDMatrixRight( l2, ip );

	CudaHelper8::calculateFirstStep<<<gpudom / threads, threads>>>(
			dt,	h, ip, matInner, matLeft, matRight,	d_w, d_u, d_z );

	CudaHelper8::synchronizeResults<<<gpudom / threads, threads>>>(
			ip,
			d_z,
			d_w );

	/*
	cudaMemcpy( z.data(), d_z, bufSize, cudaMemcpyDeviceToHost );
	for( unsigned int i = 0; i < 32; ++i )
	{
		std::cout << z[ i ] << std::endl;
	}
	*/

	solution = std::vector<double>( np );

	unsigned int kmax = ceil( T / ( nsteps * dt ) );
	error = std::vector<double>( kmax );

	for( unsigned int k = 0; k < kmax; ++k )
	{
		CudaHelper8::calculateNSteps<<<gpudom / threads, threads>>>(
				nsteps,	2.0 * ( 1.0 - l2 ),	ip,	matInner, matLeft, matRight,
				d_z, d_w, d_u );

		real* swap;
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

		CudaHelper8::synchronizeResults<<<gpudom / threads, threads>>>(
				ip,
				d_z,
				d_w );

		/*
		CudaHelper8::copyResultsLinear<<<blocks, threads>>>(
				ip, d_z, d_solution );

		cudaMemcpy( solution.data(), d_solution, np * sizeof( double ), cudaMemcpyDeviceToHost );
		*/

		cudaMemcpy(
				z.data(),
				d_z,
				bufSize,
				cudaMemcpyDeviceToHost );

		/*
		for( unsigned int i = 0; i < ns; ++i )
		{
			memcpy(
					&solution[ i * ip ],
					&z[ 2 * i * ip ],
					ip * sizeof( double ) );
		}
		*/
		for( unsigned int n = 0; n < blocks; ++n )
		{
			for( unsigned int k = 0; k < threads; ++k )
			{
				if( n == blocks - 1 && k == threads - 1 )
				{
					continue;
				}

				unsigned int base = n * threads * ip + k;
				unsigned int dstBase = ( n * threads + k ) * ip / 2;
				for( unsigned int i = 0; i < ip / 2; ++i )
				{
					solution[ dstBase + i ] = z[ base + i * threads ];
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

	cudaFree( d_z );
	cudaFree( d_w );
	cudaFree( d_u );
	gpuFreeFDMatrix( matInner );
	gpuFreeFDMatrix( matLeft );
	gpuFreeFDMatrix( matRight );
}

SpDiaMat Solver8::gpuAllocFDMatrixInner( double l2, unsigned int ip )
{
	unsigned int diagSize = ip - 1;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = 2;
	cudaMalloc( &mat.offsets, 2 * sizeof( int ) );
	cudaMalloc( &mat.values, 2 * diagSize * sizeof( real ) );

	static const int offsets[] = { 1, -1 };
	real* values = new real[ 2 * diagSize * sizeof( real ) ];

	for( unsigned int i = 0; i < diagSize; ++i )
	{
		values[ i ] = l2;
		values[ diagSize + i ] = l2;
	}

	cudaMemcpy( const_cast<int*>( mat.offsets ), offsets, 2 * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( const_cast<real*>( mat.values ), values, 2 * diagSize * sizeof( real ), cudaMemcpyHostToDevice );

	delete[] values;
	return mat;
}

SpDiaMat Solver8::gpuAllocFDMatrixLeft( double l2, unsigned int ip )
{
	unsigned int diagSize = ip - 1;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = 2;
	cudaMalloc( &mat.offsets, 2 * sizeof( int ) );
	cudaMalloc( &mat.values, 2 * diagSize * sizeof( real ) );

	static const int offsets[] = { 1, -1 };
	real* values = new real[ 2 * diagSize * sizeof( real ) ];

	for( unsigned int i = 0; i < diagSize; ++i )
	{
		values[ i ] = ( i != 0 ? l2 : 2.0 * l2 );
		values[ diagSize + i ] = l2;
	}

	cudaMemcpy( const_cast<int*>( mat.offsets ), offsets, 2 * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( const_cast<real*>( mat.values ), values, 2 * diagSize * sizeof( real ), cudaMemcpyHostToDevice );

	delete[] values;
	return mat;
}

SpDiaMat Solver8::gpuAllocFDMatrixRight( double l2, unsigned int ip )
{
	unsigned int diagSize = ip - 1;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = 2;
	cudaMalloc( &mat.offsets, 2 * sizeof( int ) );
	cudaMalloc( &mat.values, 2 * diagSize * sizeof( real ) );

	static const int offsets[] = { 1, -1 };
	real* values = new real[ 2 * diagSize * sizeof( real ) ];

	for( unsigned int i = 0; i < diagSize; ++i )
	{
		values[ i ] = l2;
		values[ diagSize + i ] = ( i != diagSize - 1 ? l2 : 2.0 * l2 );
	}

	cudaMemcpy( const_cast<int*>( mat.offsets ), offsets, 2 * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( const_cast<real*>( mat.values ), values, 2 * diagSize * sizeof( real ), cudaMemcpyHostToDevice );

	delete[] values;
	return mat;
}

void Solver8::gpuFreeFDMatrix( SpDiaMat& mat )
{
	cudaFree( const_cast<int*>( mat.offsets ) );
	mat.offsets = NULL;
	cudaFree( const_cast<real*>( mat.values ) );
	mat.values = NULL;
}
