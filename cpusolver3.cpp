#include "cpusolver3.h"
#include "cpuhelper.h"
#include <cmath>

CpuSolver3::CpuSolver3()
{
}

CpuSolver3::~CpuSolver3()
{
}


// CpuSolver3 is nearly identical to CpuSolver0 except for the lack of explicit
// matrix-vector operations
void CpuSolver3::solve()
{
	unsigned int ip = np / ns;
	real h = 2 * L / ( np - 1 );
	real dt = h / 4.0;
	real l = dt / h;
	real l2 = l * l;
	unsigned int nsteps = ip / 4;
	unsigned int kmax = ceil( T / ( nsteps * dt ) );
	//unsigned int kmax = static_cast<unsigned int>( T ) / nsteps;
	error = std::vector<double>( kmax );

	std::vector<real> x( np );
	std::vector<real> z( np );
	std::vector<real> w( np );
	std::vector<real> u( np );

	for( unsigned int i = 0; i < np; ++i )
	{
		real xi = -L + i * h;
		x[ i ] = xi;
		w[ i ] = u0( xi );
		u[ i ] = u1( xi );
	}

	real a = 1.0 - l2 - dc;
	real b = nc * dt * dt;
	calculateFirstStep( np, z.data(), w.data(), u.data(), dt, l2, a, b );

	a = 2.0 * ( 1.0 - l2 ) - dc;

	real* pz = z.data();
	real* pw = w.data();
	real* pu = u.data();
	for( unsigned int k = 0; k < kmax; ++k )
	{
		calculateNSteps( np, pz, pw, pu, l2, a, b, nsteps );

		double t = ( ( k + 1 ) * nsteps + 1 ) * dt;
		double err = 0.0;
		for( unsigned int i = 0; i < np; ++i )
		{
			double errLoc = pz[ i ] - sol( x[ i ], t );
			err += errLoc * errLoc;
		}
		error[ k ] = sqrt( h * err );
	}

	solution = std::vector<double>( np );
	for( unsigned int i = 0; i < np; ++i )
	{
		solution[ i ] = pz[ i ];
	}
}

const char* CpuSolver3::getName() const
{
	return "CpuSolver3";
}

void CpuSolver3::calculateFirstStep( unsigned int ip, real* z, const real* f,
		const real* g, real dt, real l2, real a, real b )
{
	real f3 = f[ 0 ];
	f3 *= f3 * f3;
	z[ 0 ] = l2 * f[ 1 ] + a * f[ 0 ] + dt * g[ 0 ] + b * f3;
	for( unsigned int i = 1; i < ip - 1; ++i )
	{
		f3 = f[ i ];
		f3 *= f3 * f3;
		z[ i ] = l2 * 0.5 * ( f[ i - 1 ] + f[ i + 1 ] ) + a * f[ i ] +
			dt * g[ i ] + b * f3;
	}
	f3 = f[ ip - 1 ];
	f3 *= f3 * f3;
	z[ ip - 1 ] = l2 * f[ ip - 2 ] + a * f[ ip - 1 ] + dt * g[ ip - 1 ] +
		b * f3;
}

void CpuSolver3::calculateNSteps( unsigned int ip, real*& z, real*& w, real*& u,
		real l2, real a, real b, unsigned int nsteps )
{
	real* swap;
	for( unsigned int i = 0; i < nsteps; ++i )
	{
		real z3 = z[ 0 ];
		z3 *= z3 * z3;
		u[ 0 ] = 2.0 * l2 * z[ 1 ] +
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
		u[ ip - 1 ] = 2.0 * l2 * z[ ip - 2 ] + a * z[ ip - 1 ] - w[ ip - 1 ] +
			b * z3;

		swap = w;
		w = z;
		z = u;
		u = swap;
	}
}
