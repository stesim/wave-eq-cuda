#include "solver2.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include "cudahelper2.h"
#include <iostream>

Solver2::Solver2()
{
}

Solver2::~Solver2()
{
}

void Solver2::solve()
{
	//cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

	unsigned int ip = np / ns;
	real h = 2 * L / ( np - 1 );
	real dt = h / 4.0;
	real l = dt / h;
	real l2 = l * l;
	unsigned int nsteps = ip / 4;
	unsigned int ndom = 2 * ns - 1;
	unsigned int gpudom = 2 * ns;

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
	for( unsigned int i = 0; i < ndom; ++i )
	{
		unsigned int pos = i * ip;
		memcpy( &w[ pos ], &f[ pos / 2 ], ip * sizeof( real ) );
		memcpy( &u[ pos ], &g[ pos / 2 ], ip * sizeof( real ) ); 
	}
	cudaMemcpy( d_w, w.data(), bufSize, cudaMemcpyHostToDevice );
	cudaMemcpy( d_u, u.data(), bufSize, cudaMemcpyHostToDevice );

	f.clear();
	g.clear();

	SpDiaMat matInner = gpuAllocFDMatrixInner( l2, ip );
	SpDiaMat matLeft = gpuAllocFDMatrixLeft( l2, ip );
	SpDiaMat matRight = gpuAllocFDMatrixRight( l2, ip );

	CudaHelper2::calculateFirstStep<<<gpudom / threads, threads>>>(
			dt,	h, ip, matInner, matLeft, matRight,	d_w, d_u, d_z );

	CudaHelper2::synchronizeResults<<<gpudom / threads, threads>>>(
			ip,
			d_z,
			d_w );

	solution = std::vector<double>( np );

	unsigned int kmax = ceil( T / ( nsteps * dt ) );
	error = std::vector<double>( kmax );

	for( unsigned int k = 0; k < kmax; ++k )
	{
		CudaHelper2::calculateNSteps<<<gpudom / threads, threads>>>(
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

		CudaHelper2::synchronizeResults<<<gpudom / threads, threads>>>(
				ip,
				d_z,
				d_w );

		cudaMemcpy(
				z.data(),
				d_z,
				bufSize,
				cudaMemcpyDeviceToHost );

		for( unsigned int i = 0; i < ns; ++i )
		{
			for( unsigned int j = 0; j < ip; ++j )
			{
				solution[ i * ip + j ] = z[ 2 * i * ip + j ];
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

const char* Solver2::getName() const
{
	return "Solver2";
}

SpDiaMat Solver2::gpuAllocFDMatrixInner( real l2, unsigned int ip )
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

	cudaMemcpy( mat.offsets, offsets, 2 * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( mat.values, values, 2 * diagSize * sizeof( real ), cudaMemcpyHostToDevice );

	delete[] values;
	return mat;
}

SpDiaMat Solver2::gpuAllocFDMatrixLeft( real l2, unsigned int ip )
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

	cudaMemcpy( mat.offsets, offsets, 2 * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( mat.values, values, 2 * diagSize * sizeof( real ), cudaMemcpyHostToDevice );

	delete[] values;
	return mat;
}

SpDiaMat Solver2::gpuAllocFDMatrixRight( real l2, unsigned int ip )
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

	cudaMemcpy( mat.offsets, offsets, 2 * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( mat.values, values, 2 * diagSize * sizeof( real ), cudaMemcpyHostToDevice );

	delete[] values;
	return mat;
}

void Solver2::gpuFreeFDMatrix( SpDiaMat& mat )
{
	cudaFree( mat.offsets );
	mat.offsets = NULL;
	cudaFree( mat.values );
	mat.values = NULL;
}
