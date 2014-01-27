#include "solver3.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include "cudahelper3.h"
#include <iostream>

Solver3::Solver3()
{
}

Solver3::~Solver3()
{
}

void Solver3::solve()
{
	unsigned int ip = np / ns;
	double h = 2 * L / ( np - 1 );
	double dt = h / 4.0;
	double l = dt / h;
	double l2 = l * l;
	unsigned int nsteps = ip / 2;
	unsigned int ndom = 2 * ns - 1;
	unsigned int gpudom = 2 * ns;

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
	for( unsigned int i = 0; i < ndom; ++i )
	{
		unsigned int pos = i * ip;
		memcpy( &w[ pos ], &f[ pos / 2 ], ip * sizeof( double ) );
		memcpy( &u[ pos ], &g[ pos / 2 ], ip * sizeof( double ) ); 
	}
	cudaMemcpy( d_w, w.data(), bufSize, cudaMemcpyHostToDevice );
	cudaMemcpy( d_u, u.data(), bufSize, cudaMemcpyHostToDevice );

	f.clear();
	g.clear();

	SpDiaMat matInner = gpuAllocFDMatrixInner( l2, ip );
	SpDiaMat matLeft = gpuAllocFDMatrixLeft( l2, ip );
	SpDiaMat matRight = gpuAllocFDMatrixRight( l2, ip );

	CudaHelper3::calculateFirstStep<<<gpudom / threads, threads, 3 * threads * ip * sizeof( double )>>>(
			dt,	h, ip, matInner, matLeft, matRight,	d_w, d_u, d_z );

	CudaHelper3::synchronizeResults<<<gpudom / threads, threads>>>(
			ip,
			d_z,
			d_w );

	solution = std::vector<double>( np );

	unsigned int kmax = ceil( T / ( nsteps * dt ) );
	error = std::vector<double>( kmax );

	for( unsigned int k = 0; k < kmax; ++k )
	{
		CudaHelper3::calculateNSteps<<<gpudom / threads, threads, 3 * threads * ip * sizeof( double )>>>(
				nsteps,	2.0 * ( 1.0 - l2 ),	ip,	matInner, matLeft, matRight,
				d_z, d_w, d_u );

		/*
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
		*/

		CudaHelper3::synchronizeResults<<<gpudom / threads, threads>>>(
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
			memcpy(
					&solution[ i * ip ],
					&z[ 2 * i * ip ],
					ip * sizeof( double ) );
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

SpDiaMat Solver3::gpuAllocFDMatrixInner( double l2, unsigned int ip )
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

SpDiaMat Solver3::gpuAllocFDMatrixLeft( double l2, unsigned int ip )
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

SpDiaMat Solver3::gpuAllocFDMatrixRight( double l2, unsigned int ip )
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

void Solver3::gpuFreeFDMatrix( SpDiaMat& mat )
{
	cudaFree( const_cast<int*>( mat.offsets ) );
	mat.offsets = NULL;
	cudaFree( const_cast<double*>( mat.values ) );
	mat.values = NULL;
}
