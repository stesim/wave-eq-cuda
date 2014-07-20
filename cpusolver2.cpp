#include "cpusolver2.h"
#include "cpuhelper.h"
#include <cmath>
#include <cstring>

thread_local unsigned int CpuSolver2::threadId;

CpuSolver2::CpuSolver2()
{
}

CpuSolver2::~CpuSolver2()
{
}


// CpuSolver2 is identical to CpuSolver1 except for the threadLoop method
void CpuSolver2::solve()
{
	unsigned int ip = np / ns;
	unsigned int ndom = 2 * ns - 1;
	real h = 2 * L / ( np - 1 );
	real dt = h / 4.0;
	real l = dt / h;
	real l2 = l * l;
	unsigned int nsteps = ip / 4;
	unsigned int kmax = ceil( T / ( nsteps * dt ) );
	//unsigned int kmax = static_cast<unsigned int>( T ) / nsteps;
	error = std::vector<double>( kmax );

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

	real a = 1.0 - l2 - dc;
	real b = nc * dt * dt;
	runThreadPool( true, ndom, ip, pz, pw, pu, dt, l2, a, b, nsteps );

	a = 2.0 * ( 1.0 - l2 ) - dc;
	for( unsigned int k = 0; k < kmax; ++k )
	{
		runThreadPool( false, ndom, ip, pz, pw, pu, dt, l2, a, b, nsteps );

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

	solution = std::vector<double>( np );
	for( unsigned int i = 0; i < ndom; i += 2 )
	{
		for( unsigned int j = 0; j < ip; ++j )
		{
			solution[ i / 2 * ip + j ] = pz[ i ][ j ];
		}
	}
}

const char* CpuSolver2::getName() const
{
	return "CpuSolver2";
}

void CpuSolver2::threadLoop(
		bool firstStep,
		unsigned int ndom,
		unsigned int ip,
		std::vector<real*>& pz,
		std::vector<real*>& pw,
		std::vector<real*>& pu,
		real dt,
		real l2,
		real a,
		real b,
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

		if( firstStep )
		{
			real* z = pz[ threadId ];
			real* f = pw[ threadId ];
			real* g = pu[ threadId ];

			// calculate first step
			real f3 = f[ 0 ];
			f3 *= f3 * f3;
			z[ 0 ] = ( threadId == 0 ? 2.0 * l2 : l2 ) * 0.5 * f[ 1 ] +
				a * f[ 0 ] + dt * g[ 0 ] + b * f3;
			for( unsigned int i = 1; i < ip - 1; ++i )
			{
				f3 = f[ i ];
				f3 *= f3 * f3;
				z[ i ] = l2 * 0.5 * ( f[ i - 1 ] + f[ i + 1 ] ) + a * f[ i ] +
					dt * g[ i ] + b * f3;
			}
			f3 = f[ ip - 1 ];
			f3 *= f3 * f3;
			z[ ip - 1 ] = ( threadId == ndom - 1 ? 2.0 * l2 : l2 ) *
				0.5 * f[ ip - 2 ] + a * f[ ip - 1 ] + dt * g[ ip - 1 ] + b * f3;
		}
		else
		{
			real*& z = pz[ threadId ];
			real*& w = pw[ threadId ];
			real*& u = pu[ threadId ];

			// calculate 'nsteps' time steps
			for( unsigned int n = 0; n < nsteps; ++n )
			{
				real z3 = z[ 0 ];
				z3 *= z3 * z3;
				u[ 0 ] = ( threadId == 0 ? 2.0 * l2 : l2 ) * z[ 1 ] +
					a * z[ 0 ] - w[ 0 ] + b * z3;
				for( unsigned int i = 1; i < ip - 1; ++i )
				{
					z3 = z[ i ];
					z3 *= z3 * z3;
					u[ i ] = l2 * ( z[ i - 1 ] + z[ i + 1 ] ) +
						a * z[ i ] - w[ i ] + b * z3;
				}
				z3 = z[ ip - 1 ];
				z3 *= z3 * z3;
				u[ ip - 1 ] = ( threadId == ndom - 1 ? 2.0 * l2 : l2 ) *
					z[ ip - 2 ] + a * z[ ip - 1 ] - w[ ip - 1 ] + b * z3;

				real* swap = w;
				w = z;
				z = u;
				u = swap;
			}
		}
	}
}

void CpuSolver2::runThreadPool(
		bool firstStep,
		unsigned int ndom,
		unsigned int ip,
		std::vector<real*>& pz,
		std::vector<real*>& pw,
		std::vector<real*>& pu,
		real dt,
		real l2,
		real a,
		real b,
		unsigned int nsteps )
{
	std::vector<std::thread> vecThreads( threads );
	m_uiThreadId = 0;
	for( unsigned int i = 0; i < threads; ++i )
	{
		vecThreads[ i ] = std::thread( &CpuSolver2::threadLoop, this,
				firstStep, ndom, ip, std::ref( pz ), std::ref( pw ),
				std::ref( pu ), dt, l2, a, b, nsteps );
	}
	for( unsigned int i = 0; i < threads; ++i )
	{
		vecThreads[ i ].join();
	}
}
