#include "solver6.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include "cudahelper6.h"
#include <iostream>

Solver6::Solver6()
{
}

Solver6::~Solver6()
{
}

void Solver6::solve()
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

	double* d_z;
	double* d_w;
	double* d_u;
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
	/*
	for( unsigned int i = 0; i < ndom; ++i )
	{
		unsigned int pos = i * ip;
		memcpy( &w[ pos ], &f[ pos / 2 ], ip * sizeof( double ) );
		memcpy( &u[ pos ], &g[ pos / 2 ], ip * sizeof( double ) ); 
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
	for( unsigned int i = 0; i < 32; ++i )
	{
		std::cout << w[ i ] << std::endl;
	}
	*/
	/*
	double* d_f;
	double* d_g;
	cudaMalloc( &d_f, np * sizeof( double ) );
	cudaMalloc( &d_g, np * sizeof( double ) );
	cudaMemcpy( d_f, f.data(), np * sizeof( double ), cudaMemcpyHostToDevice );
	cudaMemcpy( d_g, g.data(), np * sizeof( double ), cudaMemcpyHostToDevice );
	CudaHelper6::copyInitialValuesCoalesced<<<blocks, threads>>>(
			ip, d_f, d_g, d_w, d_u );
	cudaFree( d_f );
	double* d_solution = d_g;
	*/

	cudaMemcpy( d_w, w.data(), bufSize, cudaMemcpyHostToDevice );
	cudaMemcpy( d_u, u.data(), bufSize, cudaMemcpyHostToDevice );

	f.clear();
	g.clear();

	SpDiaMat matInner = gpuAllocFDMatrixInner( l2, ip );
	SpDiaMat matLeft = gpuAllocFDMatrixLeft( l2, ip );
	SpDiaMat matRight = gpuAllocFDMatrixRight( l2, ip );

	CudaHelper6::calculateFirstStep<<<gpudom / threads, threads>>>(
			dt,	h, ip, matInner, matLeft, matRight,	d_w, d_u, d_z );

	CudaHelper6::synchronizeResults<<<gpudom / threads, threads>>>(
			ip,
			d_z,
			d_w );

		/*
		cudaMemcpy( z.data(), d_z, bufSize, cudaMemcpyDeviceToHost );
		for( unsigned int n = 0; n < blocks; ++n )
		{
			for( unsigned int j = 0; j < threads; j += 2 )
			{
				if( n == blocks - 1 && j == threads - 1 )
				{
					continue;
				}

				unsigned int base = n * threads * ip + j;
				unsigned int dstBase = ( n * threads + j ) * ip / 2;
				for( unsigned int i = 0; i < ip; ++i )
				{
					solution[ dstBase + i ] = z[ base + i * threads ];
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
		CudaHelper6::calculateNSteps<<<gpudom / threads, threads>>>(
				nsteps,	2.0 * ( 1.0 - l2 ),	ip,	matInner, matLeft, matRight,
				d_z, d_w, d_u );

		double* swap;
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

		CudaHelper6::synchronizeResults<<<gpudom / threads, threads>>>(
				ip,
				d_z,
				d_w );

		/*
		CudaHelper6::copyResultsLinear<<<blocks, threads>>>(
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
			for( unsigned int j = 0; j < threads; j += 2 )
			{
				if( n == blocks - 1 && j == threads - 1 )
				{
					continue;
				}

				unsigned int base = n * threads * ip + j;
				unsigned int dstBase = ( n * threads + j ) * ip / 2;
				for( unsigned int i = 0; i < ip; ++i )
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

	solution = z;

	cudaFree( d_z );
	cudaFree( d_w );
	cudaFree( d_u );
	gpuFreeFDMatrix( matInner );
	gpuFreeFDMatrix( matLeft );
	gpuFreeFDMatrix( matRight );
}

const char* Solver6::getName() const
{
	return "Solver6";
}

SpDiaMat Solver6::gpuAllocFDMatrixInner( double l2, unsigned int ip )
{
	unsigned int diagSize = ip - 1;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = 2;
	cudaMalloc( &mat.offsets, 2 * sizeof( int ) );
	cudaMalloc( &mat.values, 2 * diagSize * sizeof( double ) );

	static const int offsets[] = { 1, -1 };
	double* values = new double[ 2 * diagSize * sizeof( double ) ];

	for( unsigned int i = 0; i < diagSize; ++i )
	{
		values[ i ] = l2;
		values[ diagSize + i ] = l2;
	}

	cudaMemcpy( const_cast<int*>( mat.offsets ), offsets, 2 * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( const_cast<double*>( mat.values ), values, 2 * diagSize * sizeof( double ), cudaMemcpyHostToDevice );

	delete[] values;
	return mat;
}

SpDiaMat Solver6::gpuAllocFDMatrixLeft( double l2, unsigned int ip )
{
	unsigned int diagSize = ip - 1;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = 2;
	cudaMalloc( &mat.offsets, 2 * sizeof( int ) );
	cudaMalloc( &mat.values, 2 * diagSize * sizeof( double ) );

	static const int offsets[] = { 1, -1 };
	double* values = new double[ 2 * diagSize * sizeof( double ) ];

	for( unsigned int i = 0; i < diagSize; ++i )
	{
		values[ i ] = ( i != 0 ? l2 : 2.0 * l2 );
		values[ diagSize + i ] = l2;
	}

	cudaMemcpy( const_cast<int*>( mat.offsets ), offsets, 2 * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( const_cast<double*>( mat.values ), values, 2 * diagSize * sizeof( double ), cudaMemcpyHostToDevice );

	delete[] values;
	return mat;
}

SpDiaMat Solver6::gpuAllocFDMatrixRight( double l2, unsigned int ip )
{
	unsigned int diagSize = ip - 1;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = 2;
	cudaMalloc( &mat.offsets, 2 * sizeof( int ) );
	cudaMalloc( &mat.values, 2 * diagSize * sizeof( double ) );

	static const int offsets[] = { 1, -1 };
	double* values = new double[ 2 * diagSize * sizeof( double ) ];

	for( unsigned int i = 0; i < diagSize; ++i )
	{
		values[ i ] = l2;
		values[ diagSize + i ] = ( i != diagSize - 1 ? l2 : 2.0 * l2 );
	}

	cudaMemcpy( const_cast<int*>( mat.offsets ), offsets, 2 * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( const_cast<double*>( mat.values ), values, 2 * diagSize * sizeof( double ), cudaMemcpyHostToDevice );

	delete[] values;
	return mat;
}

void Solver6::gpuFreeFDMatrix( SpDiaMat& mat )
{
	cudaFree( const_cast<int*>( mat.offsets ) );
	mat.offsets = NULL;
	cudaFree( const_cast<double*>( mat.values ) );
	mat.values = NULL;
}
