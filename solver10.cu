#include "solver10.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include "cudahelper10.h"
#include <iostream>

Solver10::Solver10()
{
}

Solver10::~Solver10()
{
}

void Solver10::solve()
{
	//cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
	cudaDeviceSetCacheConfig( cudaFuncCachePreferShared );

	unsigned int ip = np / ns;
	real h = 2 * L / ( np - 1 );
	real dt = h / 4.0;
	real l = dt / h;
	real l2 = l * l;
	unsigned int nsteps = ip / 2;
	unsigned int ndom = 2 * ns - 1;
	unsigned int gpudom = 2 * ns;
	unsigned int blocks = gpudom / threads;

	unsigned int kmax = ceil( T / ( nsteps * dt ) );
	solution = std::vector<double>( np );
	error = std::vector<double>( kmax );

	std::vector<real> z( gpudom * ip );
	std::vector<real> w( gpudom * ip );
	std::vector<real> u( gpudom * ip );

	unsigned int bufSize = gpudom * ip * sizeof( real );

	real2* d_z;
	real2* d_w;
	real2* d_u;
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

	CudaHelper10::calculateFirstStep<<<blocks, threads>>>(
			dt,	h, ip / 2, d_w, d_u, d_z );

	CudaHelper10::synchronizeResults<<<blocks, threads>>>(
			ip / 2,
			d_z,
			d_w );

	for( unsigned int k = 0; k < kmax; ++k )
	{
		CudaHelper10::calculateNSteps<<<blocks, threads>>>(
				nsteps,	l2, 2.0 * ( 1.0 - l2 ), ip / 2, d_z, d_w, d_u );

		real2* swap;
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

		CudaHelper10::synchronizeResults<<<blocks, threads>>>(
				ip / 2,
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

	cudaFree( d_z );
	cudaFree( d_w );
	cudaFree( d_u );
}

const char* Solver10::getName() const
{
	return "Solver10";
}
