#include <iostream>
#include "solver6.h"
#include "solver7.h"
#include "solver9.h"
#include <chrono>
#include <fstream>

double u0( double x )
{
	return 1.0 / ( 1.0 + x * x );
}

double u1( double x )
{
	return 0.0;
}

double sol( double x, double t )
{
	double a = x - t;
	double b = x + t;
	return ( 1.0 / ( 1.0 + a * a ) + 1.0 / ( 1.0 + b * b ) ) / 2.0;
}

int main( int argc, char* argv[] )
{
#ifdef VAL_N
	unsigned int N = VAL_N;
#else
	unsigned int N = 14;
#endif
#ifdef VAL_n
	unsigned int n = VAL_n;
#else
	unsigned int n = 7;
#endif
	double L = 150.0;
	double T = 100.0;
	
	unsigned int threadsMin = 16;
	unsigned int threadsMax = 64;
	unsigned int cases = 0;
	for( unsigned int i = threadsMin; i <= threadsMax; i <<= 1 )
	{
		++cases;
	}

	std::vector<SolverBase*> solvers( 3 );
	solvers[ 0 ] = new Solver6();
	solvers[ 1 ] = new Solver7();
	solvers[ 2 ] = new Solver9();
	for( unsigned int i = 0; i < solvers.size(); ++i )
	{
		SolverBase& s = *solvers[ i ];
		s.np = ( 1 << N );
		s.ns = ( 1 << n );
		s.L = L;
		s.T = T;
		s.u0 = u0;
		s.u1 = u1;
		s.sol = sol;
	}
		
	std::vector<unsigned int> threads( cases * solvers.size() );
	std::vector<double> time( cases * solvers.size() );
	std::vector<double> error( cases * solvers.size() );

	unsigned int runs = 2;
	
	char filename[ 256 ];
	for( unsigned int i = 0; i < runs; ++i )
	{
		for( unsigned int j = 0; j < solvers.size(); ++j )
		{
			SolverBase& s = *solvers[ j ];
			std::cout << "solver: " << s.getName() << std::endl;

			for( unsigned int k = 0; k < cases; ++k )
			{
				s.threads = ( threadsMin << k );
				std::cout << "threads: " << s.threads << std::endl;

				auto start = std::chrono::high_resolution_clock::now();
				s.solve();
				auto elapsed = std::chrono::high_resolution_clock::now() - start;
				double realTime = (double)elapsed.count()
					* std::chrono::high_resolution_clock::period::num
					/ std::chrono::high_resolution_clock::period::den;
				std::cout << "time: " << realTime << 's' << std::endl;
				std::cout << "error: " << s.error.back() << std::endl;
				std::cout << std::endl;

				unsigned int resIdx = j * cases + k;
				threads[ resIdx ] = s.threads;
				time[ resIdx ] = realTime;
				error[ resIdx ] = s.error.back();
			}
		}

		sprintf( filename, "results-N%i-n%i-run%i.txt", N, n, i );
		std::ofstream file( filename );
		file << "N n L T" << std::endl;
		file << N << ' ' << n << ' ' << L << ' ' << T << std::endl;
		file << std::endl;
		file << "solver threads time error" << std::endl;
		for( unsigned int j = 0; j < solvers.size(); ++j )
		{
			for( unsigned int k = 0; k < cases; ++k )
			{
				unsigned int idx = j * cases + k;
				file << solvers[ j ]->getName() << ' ' << threads[ idx ] << ' '
					<< time[ idx ] << ' ' << error[ idx ] << std::endl;

			}
		}
		file.close();
	}

	for( unsigned int i = 0; i < solvers.size(); ++i )
	{
		delete solvers[ i ];
	}

	return 0;
}
