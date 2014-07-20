#include "cpusolver0.h"
#include "cpuhelper.h"
#include <cmath>

CpuSolver0::CpuSolver0()
{
}

CpuSolver0::~CpuSolver0()
{
}


void CpuSolver0::solve()
{
	// for detailed explanation of the solve method see Solver11::solve

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

	SpDiaMat mat = allocFDMatrix( l2, np );

	real a = 1.0 - l2 - dc;
	real b = nc * dt * dt;
	calculateFirstStep( np, z.data(), w.data(), u.data(), mat, dt, a, b );

	a = 2.0 * ( 1.0 - l2 ) - dc;

	real* pz = z.data();
	real* pw = w.data();
	real* pu = u.data();
	for( unsigned int k = 0; k < kmax; ++k )
	{
		calculateNSteps( np, pz, pw, pu, mat, a, b, nsteps );

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

const char* CpuSolver0::getName() const
{
	return "CpuSolver0";
}

void CpuSolver0::calculateFirstStep(
		unsigned int ip, real* z, const real* f, const real* g,
		const SpDiaMat& mat, real dt, real a, real b )
{
	CpuHelper::mulMatVec( mat, f, z );
	for( unsigned int i = 0; i < ip; ++i )
	{
		double f3 = f[ i ];
		f3 *= f3 * f3;
		z[ i ] = 0.5 * z[ i ] + a * f[ i ] + dt * g[ i ] + b * f3;
	}
}

void CpuSolver0::calculateNSteps( unsigned int ip, real*& z, real*& w, real*& u,
		const SpDiaMat& mat, real a, real b, unsigned int nsteps )
{
	real* swap;
	for( unsigned int i = 0; i < nsteps; ++i )
	{
		CpuHelper::mulMatVec( mat, z, u );
		for( unsigned int i = 0; i < ip; ++i )
		{
			double z3 = z[ i ];
			z3 *= z3 * z3;
			u[ i ] += a * z[ i ] - w[ i ] + b * z3;
		}

		swap = w;
		w = z;
		z = u;
		u = swap;
	}
}

SpDiaMat CpuSolver0::allocFDMatrix( real l2, unsigned int ip )
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
		mat.values[ diagSize + i ] = ( i != diagSize - 1 ? l2 : 2.0 * l2 );
	}

	return mat;
}
