#include "solver11.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include "cudahelper11.h"
#include <iostream>

/*
 * Create two-dimensional dim3-structure to start a greater number of threads
 * than are supported in a single dimension.
*/
static dim3 createDim2DFrom1D( unsigned int size, unsigned int maxSize )
{
	dim3 dim;
	if( size > maxSize )
	{
		dim.x = maxSize;
		dim.y = size / maxSize;
	}
	else
	{
		dim.x = size;
		dim.y = 1;
	}
	dim.z = 1;

	return dim;
}

Solver11::Solver11()
{
}

Solver11::~Solver11()
{
}

void Solver11::solve()
{
	// Set maximum shared memory size to 48kB
	cudaDeviceSetCacheConfig( cudaFuncCachePreferShared );

	unsigned int ip = np / ns;			// subdomain size
	real h = 2 * L / ( np - 1 );		// spacial step size
	real dt = h / 4.0;					// temporal step size
	real dt2 = dt * dt;
	real l = dt / h;
	real l2 = l * l;
	unsigned int nsteps = ip / 4;		// number of steps per cycle
	unsigned int ndom = 2 * ns - 1;		// number of overlapping subdomains
	unsigned int gpudom = 2 * ns;		// even number of domains for gpu

	// calculate grid / block sizes
	dim3 gridSize = createDim2DFrom1D( gpudom, 1 << 16 );
	unsigned int threads = ip / 2;
	dim3 blockSize = createDim2DFrom1D( threads, 1024 );
	unsigned int syncThreads = threads / 4;
	dim3 syncBlockSize = createDim2DFrom1D( syncThreads, 1024 );
	unsigned int shmem = ( ip + 4 ) * sizeof( real );

	unsigned int kmax = ceil( T / ( nsteps * dt ) );	// number of cycles
	//unsigned int kmax = static_cast<unsigned int>( T ) / nsteps;
	solution = std::vector<double>( np );
	error = std::vector<double>( kmax );

	// size of the solution vectors in bytes
	unsigned int bufSize = gpudom * ip * sizeof( real );

	// allocate necessary host memory
	real* z;
	real* w;
	cudaMallocHost( &z, bufSize );
	cudaMallocHost( &w, bufSize );

	// allocate necessary device memory
	real2* d_z;
	real2* d_w;
	cudaMalloc( &d_z, bufSize );
	cudaMalloc( &d_w, bufSize );

	// write initial values to host memory
	std::vector<real> x( np );
	std::vector<real> f( np );
	std::vector<real> g( np );
	for( unsigned int i = 0; i < np; ++i )
	{
		x[ i ] = -L + i * h;
		f[ i ] = u0( x[ i ] );
		g[ i ] = u1( x[ i ] );
	}
	for( unsigned int n = 0; n < ndom; ++n )
	{
		memcpy( &w[ n * ip ], &f[ n * ip / 2 ], ip * sizeof( real ) );
		memcpy( &z[ n * ip ], &g[ n * ip / 2 ], ip * sizeof( real ) );
	}

	// copy to device memory
	cudaMemcpy( d_w, w, bufSize, cudaMemcpyHostToDevice );
	cudaMemcpy( d_z, z, bufSize, cudaMemcpyHostToDevice );

	f.clear();
	g.clear();

	real a = 1.0 - l2 - dc;		// linear coefficient for first step
	real b = nc * dt2;			// cubic coefficient

	// calculate first time step separately using the initial values
	CudaHelper11::calculateFirstStep<<<gridSize, blockSize, shmem>>>(
			dt,	h, l2, a, b, ip / 2, d_w, d_z, d_z );
	// replace invalid values with the corresponding correct values
	CudaHelper11::synchronizeResults<<<gridSize, syncBlockSize>>>(
			ip / 2, d_z, d_w );
	
	a = 2.0 * ( 1.0 - l2 ) - dc;	// linear coefficient for remaining steps
	for( unsigned int k = 0; k < kmax; ++k )
	{
		// calculate 'nsteps' time steps
		CudaHelper11::calculateNSteps<<<gridSize, blockSize, shmem>>>(
				nsteps,	l2, a, b, ip / 2, d_z, d_w );
		CudaHelper11::synchronizeResults<<<gridSize, syncBlockSize>>>(
				ip / 2, d_z, d_w );

		// calculate the error of the previous time step while the current
		// iteration is being computed on the gpu
		if( k > 0 )
		{
			calculateError( (real*)z, k - 1, ip, nsteps, dt, h, x.data() );
		}

		// wait for the gpu computations to finish and copy the results back
		// to host memory
		cudaMemcpy( z, d_z, bufSize, cudaMemcpyDeviceToHost );
	}

	// compute last iteration's error
	calculateError( (real*)z, kmax - 1, ip, nsteps, dt, h, x.data() );

	// free allocated memory
	cudaFree( d_z );
	cudaFree( d_w );

	cudaFreeHost( z );
	cudaFreeHost( w );
}

const char* Solver11::getName() const
{
	return "Solver11";
}

/*
   Calculate the L2-error of the numerical solution.
*/
void Solver11::calculateError( real* z, unsigned int k,	unsigned int ip,
		unsigned int nsteps, real dt, real h, real* x )
{
	// join the solutions on the subdomains to a global solution
	for( unsigned int n = 0; n < ns; ++n )
	{
		memcpy( &solution[ n * ip ], &z[ 2 * n * ip ], ip * sizeof( real ) );
	}

	// calculate error
	double t = ( ( k + 1 ) * nsteps + 1 ) * dt;
	double l2err = 0.0;
	for( unsigned int i = 0; i < np; ++i )
	{
		double err = sol( x[ i ], t ) - solution[ i ];
		l2err += err * err;
	}

	error[ k ] = sqrt( h * l2err );
}
