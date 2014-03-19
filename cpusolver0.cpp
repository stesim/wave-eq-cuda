#include "cpusolver0.h"
#include "cpuhelper.h"
#include <cmath>

#include <iostream>

CpuSolver0::CpuSolver0()
{
}

CpuSolver0::~CpuSolver0()
{
}


void CpuSolver0::solve()
{
	unsigned int ip = np / ns;
	real h = 2 * L / ( np - 1 );
	real dt = h / 4.0;
	real l = dt / h;
	real l2 = l * l;
	unsigned int nsteps = ip / 2;
	unsigned int kmax = ceil( T / ( nsteps * dt ) );
	error = std::vector<real>( kmax );

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

	calculateFirstStep( np, z.data(), w.data(), u.data(), mat, dt, l2 );

	real* pz = z.data();
	real* pw = w.data();
	real* pu = u.data();
	std::cout << kmax << std::endl;
	for( unsigned int k = 0; k < kmax; ++k )
	{
		calculateNSteps( np, pz, pw, pu, mat, l2, nsteps );

		double t = ( ( k + 1 ) * nsteps + 1 ) * dt;
		double err = 0.0;
		for( unsigned int i = 0; i < np; ++i )
		{
			double errLoc = pz[ i ] - sol( x[ i ], t );
			err += errLoc * errLoc;
		}
		error[ k ] = sqrt( h * err );
	}

	solution = std::vector<real>( np );
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
		unsigned int ip,
		real* z,
		const real* f,
		const real* g,
		const SpDiaMat& mat,
		real dt,
		real l2 )
{
	CpuHelper::mulMatVec( mat, f, z );
	CpuHelper::addScaledVecs( ip, 0.5, z, 1.0 - l2,	f, z );
	CpuHelper::addVecScaledVec( ip, z, dt, g );
}

void CpuSolver0::calculateNSteps(
		unsigned int ip,
		real*& z,
		real*& w,
		real*& u,
		const SpDiaMat& mat,
		real l2,
		unsigned int nsteps )
{
	real a = 2.0 * ( 1.0 - l2 );
	real* swap;
	for( unsigned int i = 0; i < nsteps; ++i )
	{
		CpuHelper::mulMatVec( mat, z, u );
		CpuHelper::addVecScaledVec( ip, u, a, z );
		CpuHelper::addVecScaledVec( ip, u, -1.0, w );

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
