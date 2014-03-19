#include "cpusolver1.h"
#include "cpuhelper.h"
#include <cmath>
#include <cstring>

#include <iostream>

thread_local unsigned int CpuSolver1::threadId;

CpuSolver1::CpuSolver1()
{
}

CpuSolver1::~CpuSolver1()
{
}


void CpuSolver1::solve()
{
	unsigned int ip = np / ns;
	unsigned int ndom = 2 * ns - 1;
	real h = 2 * L / ( np - 1 );
	real dt = h / 4.0;
	real l = dt / h;
	real l2 = l * l;
	unsigned int nsteps = ip / 2;
	unsigned int kmax = ceil( T / ( nsteps * dt ) );
	error = std::vector<real>( kmax );

	std::vector<real> x( np );
	std::vector<real> f( np );
	std::vector<real> g( np );

	for( unsigned int i = 0; i < np; ++i )
	{
		real xi = -L + i * h;
		x[ i ] = xi;
		f[ i ] = u0( xi );
		g[ i ] = u1( xi );
	}

	std::vector<real> z( ndom * ip );
	std::vector<real> w( ndom * ip );
	std::vector<real> u( ndom * ip );

	std::vector<real*> pz( ndom );
	std::vector<real*> pw( ndom );
	std::vector<real*> pu( ndom );

	for( unsigned int i = 0; i < ndom; ++i )
	{
		pz[ i ] = &z[ i * ip ];
		pw[ i ] = &w[ i * ip ];
		pu[ i ] = &u[ i * ip ];

		memcpy( pw[ i ], &f[ i * ip / 2 ], ip * sizeof( real ) );
		memcpy( pu[ i ], &g[ i * ip / 2 ], ip * sizeof( real ) );
	}

	SpDiaMat matInner = allocFDMatrixInner( l2, ip );
	SpDiaMat matLeft = allocFDMatrixLeft( l2, ip );
	SpDiaMat matRight = allocFDMatrixRight( l2, ip );

	runThreadPool( true, ndom, ip, pz, pw, pu, matInner, matLeft, matRight,
			dt, l2, nsteps );

	for( unsigned int k = 0; k < kmax; ++k )
	{
		runThreadPool( false, ndom, ip, pz, pw, pu, matInner, matLeft, matRight,
				dt, l2, nsteps );

		memcpy( &pz[ 0 ][ ip * 3 / 4 ], &pz[ 1 ][ ip / 4 ],
				ip / 4 * sizeof( real ) );
		memcpy( &pw[ 0 ][ ip * 3 / 4 ], &pw[ 1 ][ ip / 4 ],
				ip / 4 * sizeof( real ) );
		for( unsigned i = 1; i < ndom - 1; ++i )
		{
			memcpy( &pz[ i ][ 0 ], &pz[ i - 1 ][ ip / 2 ],
					ip / 4 * sizeof( real ) );
			memcpy( &pw[ i ][ 0 ], &pw[ i - 1 ][ ip / 2 ],
					ip / 4 * sizeof( real ) );

			memcpy( &pz[ i ][ ip * 3 / 4 ], &pz[ i + 1 ][ ip / 4 ],
					ip / 4 * sizeof( real ) );
			memcpy( &pw[ i ][ ip * 3 / 4 ], &pw[ i + 1 ][ ip / 4 ],
					ip / 4 * sizeof( real ) );
		}
		memcpy( &pz[ ndom - 1 ][ 0 ], &pz[ ndom - 2 ][ ip / 2 ],
				ip / 4 * sizeof( real ) );
		memcpy( &pw[ ndom - 1 ][ 0 ], &pw[ ndom - 2 ][ ip / 2 ],
				ip / 4 * sizeof( real ) );

		double t = ( ( k + 1 ) * nsteps + 1 ) * dt;
		double err = 0.0;
		for( unsigned int i = 0; i < ndom; i += 2 )
		{
			for( unsigned int j = 0; j < ip; ++j )
			{
				double errLoc = pz[ i ][ j ] - sol( x[ i / 2 * ip + j ], t );
				err += errLoc * errLoc;
			}
		}
		error[ k ] = sqrt( h * err );
	}

	solution = std::vector<real>( np );
	for( unsigned int i = 0; i < ndom; i += 2 )
	{
		for( unsigned int j = 0; j < ip; ++j )
		{
			solution[ i / 2 * ip + j ] = pz[ i ][ j ];
		}
	}
}

const char* CpuSolver1::getName() const
{
	return "CpuSolver1";
}

SpDiaMat CpuSolver1::allocFDMatrixInner( real l2, unsigned int ip )
{
	unsigned int diagSize = ip - 1;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = 2;
	mat.offsets = new int[ mat.diags ];
	mat.values = new real[ mat.diags * diagSize ];

	mat.offsets[ 0 ] = 1;
	mat.offsets[ 1 ] = -1;

	for( unsigned int i = 0; i < diagSize; ++i )
	{
		mat.values[ i ] = l2;
		mat.values[ diagSize + i ] = l2;
	}

	return mat;
}

SpDiaMat CpuSolver1::allocFDMatrixLeft( real l2, unsigned int ip )
{
	unsigned int diagSize = ip - 1;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = 2;
	mat.offsets = new int[ mat.diags ];
	mat.values = new real[ mat.diags * diagSize ];

	mat.offsets[ 0 ] = 1;
	mat.offsets[ 1 ] = -1;

	for( unsigned int i = 0; i < diagSize; ++i )
	{
		mat.values[ i ] = ( i != 0 ? l2 : 2.0 * l2 );
		mat.values[ diagSize + i ] = l2;
	}

	return mat;
}

SpDiaMat CpuSolver1::allocFDMatrixRight( real l2, unsigned int ip )
{
	unsigned int diagSize = ip - 1;

	SpDiaMat mat;
	mat.n = ip;
	mat.diags = 2;
	mat.offsets = new int[ mat.diags ];
	mat.values = new real[ mat.diags * diagSize ];

	mat.offsets[ 0 ] = 1;
	mat.offsets[ 1 ] = -1;

	for( unsigned int i = 0; i < diagSize; ++i )
	{
		mat.values[ i ] = l2;
		mat.values[ diagSize + i ] = ( i != diagSize - 1 ? l2 : 2.0 * l2 );
	}

	return mat;
}

void CpuSolver1::threadLoop(
		bool firstStep,
		unsigned int ndom,
		unsigned int ip,
		std::vector<real*>& pz,
		std::vector<real*>& pw,
		std::vector<real*>& pu,
		const SpDiaMat& matInner,
		const SpDiaMat& matLeft,
		const SpDiaMat& matRight,
		real dt,
		real l2,
		unsigned int nsteps )
{
	while( true )
	{
		m_Mutex.lock();
		threadId = m_uiThreadId++;
		m_Mutex.unlock();

		if( threadId >= ndom )
		{
			break;
		}

		const SpDiaMat& mat = ( threadId == 0 ? matLeft
				: ( threadId == ndom - 1 ? matRight : matInner ) );

		if( firstStep )
		{
			calculateFirstStep( ip, pz[ threadId ], pw[ threadId ],
					pu[ threadId ],	mat, dt, l2 );
		}
		else
		{
			calculateNSteps( ip, pz[ threadId ], pw[ threadId ], pu[ threadId ],
					mat, l2, nsteps );
		}
	}
}

void CpuSolver1::runThreadPool(
		bool firstStep,
		unsigned int ndom,
		unsigned int ip,
		std::vector<real*>& pz,
		std::vector<real*>& pw,
		std::vector<real*>& pu,
		const SpDiaMat& matInner,
		const SpDiaMat& matLeft,
		const SpDiaMat& matRight,
		real dt,
		real l2,
		unsigned int nsteps )
{
	std::vector<std::thread> vecThreads( threads );
	m_uiThreadId = 0;
	for( unsigned int i = 0; i < threads; ++i )
	{
		vecThreads[ i ] = std::thread( &CpuSolver1::threadLoop, this,
				firstStep, ndom, ip, std::ref( pz ), std::ref( pw ),
				std::ref( pu ), std::cref( matInner ), std::cref( matLeft ),
				std::cref( matRight ), dt, l2, nsteps );
	}
	for( unsigned int i = 0; i < threads; ++i )
	{
		vecThreads[ i ].join();
	}
}
