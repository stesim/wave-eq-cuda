#include "solver.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cudahelper.h"
#include <iostream>

Solver::Solver()
{
}

Solver::~Solver()
{
}

void Solver::solve()
{

	cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
	
	unsigned int ip = np;
	double h = 2 * L / ( np - 1 );
	double dt = h / 4.0;
	double l = dt / h;
	double l2 = l * l;
	unsigned int nsteps = np / ns / 4;

	std::vector<double> z( np );
	std::vector<double> w( np );
	std::vector<double> u( np );

	unsigned int bufSize = np * sizeof( double );

	double* d_z;
	double* d_w;
	double* d_u;
	cudaMalloc( &d_z, bufSize );
	cudaMalloc( &d_w, bufSize );
	cudaMalloc( &d_u, bufSize );

	for( unsigned int i = 0; i < np; ++i )
	{
		double x = -L + 2.0 * L * (double)i / (np - 1);
		w[ i ] = u0( x );
		u[ i ] = u1( x );
	}

	cudaMemcpy( d_w, w.data(), bufSize, cudaMemcpyHostToDevice );
	cudaMemcpy( d_u, u.data(), bufSize, cudaMemcpyHostToDevice );

	std::vector<int> offsets;
	std::vector<double> values;
	generateFDMatrix( l2, ip, offsets, values );
	unsigned int diags = offsets.size();

	int* d_offsets;
	double* d_values;
	cudaMalloc( &d_offsets, diags * sizeof( int ) );
	cudaMalloc( &d_values, values.size() * sizeof( double ) );

	cudaMemcpy(
			d_offsets,
			offsets.data(),
			offsets.size() * sizeof( int ),
			cudaMemcpyHostToDevice );
	cudaMemcpy(
			d_values,
			values.data(),
			values.size() * sizeof( double ),
			cudaMemcpyHostToDevice );

	CudaHelper::calculateFirstStep<<<np / threads, threads>>>(
			dt,
			h,
			ip,
			diags,
			d_offsets,
			d_values,
			d_w,
			d_u,
			d_z );

	unsigned int kmax = ceil( T / ( nsteps * dt ) );
	error = std::vector<double>( kmax );

	for( unsigned int k = 0; k < kmax; ++k )
	{
		for( unsigned int i = 0; i < nsteps; ++i )
		{
			CudaHelper::calculateNextStep<<<np / threads, threads>>>(
					ip,
					2.0 * ( 1.0 - l2 ),
					diags,
					d_offsets,
					d_values,
					d_z,
					d_w,
					d_u );

			double* swap = d_w;
			d_w = d_z;
			d_z = d_u;
			d_u = swap;
		}

		cudaMemcpy(
				z.data(),
				d_z,
				bufSize,
				cudaMemcpyDeviceToHost );

		double t = ( ( k + 1 ) * nsteps + 1 ) * dt;
		double l2err = 0.0;
		for( unsigned int i = 0; i < np; ++i )
		{
			double x = -L + 2.0 * L * (double)i / (np - 1);
			double err = sol( x, t ) - z[ i ];
			l2err += err * err;
		}
		error[ k ] = sqrt( h * l2err );
	}

	solution = z;

	cudaFree( d_z );
	cudaFree( d_w );
	cudaFree( d_u );
	cudaFree( d_offsets );
	cudaFree( d_values );
}

void Solver::generateFDMatrix(
		double l2,
		unsigned int ip,
		std::vector<int>& offsets,
		std::vector<double>& values )
{
	offsets = std::vector<int>( 2 );
	offsets[ 0 ] = 1;
	offsets[ 1 ] = -1;

	values = std::vector<double>( 2 * ( ip - 1 ), l2 );
	values[ 0 ] = 2.0 * l2;
	values[ values.size() - 1 ] = 2.0 * l2;
}

const char* Solver::getName() const
{
	return "Solver";
}
