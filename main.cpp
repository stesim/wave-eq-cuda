#include <iostream>
#include "solverbase.h"
#include "cpusolver0.h"
#include "cpusolver1.h"
#include "cpusolver2.h"
#include "solver2.h"
#include "solver6.h"
#include "solver7.h"
#include "solver9.h"
#include "solver10.h"
#include "solver11.h"
#include <chrono>
#include <fstream>
#include <vector>
#include <cstring>

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

inline bool inputUInt( const char* str, unsigned int& val )
{
	val = atoi( str );
	return ( val != 0 );
}

inline void printParams( const SolverBase& solver )
{
	std::cout << "Parameter overview:" << std::endl;
	std::cout << "\tL:\t\t" << solver.L << std::endl;
	std::cout << "\tT:\t\t" << solver.T << std::endl;
	std::cout << "\tnp:\t\t" << solver.np << std::endl;
	std::cout << "\tns:\t\t" << solver.ns << std::endl;
	std::cout << "\tSolver:\t\t" << solver.getName() << std::endl;
	std::cout << "\tThreads:\t" << solver.threads << std::endl;
}

int main( int argc, char* argv[] )
{
	double L = 150.0;
	double T = 100.0;
	
	unsigned int N;
	unsigned int n;
	unsigned int t;

	if( argc != 5 )
	{
		std::cout << "Wrong parameter count." << std::endl;
		return EXIT_FAILURE;
	}

	SolverBase* s;
	if( strcmp( argv[ 1 ], "cpu0" ) == 0 )
	{
		s = new CpuSolver0();
	}
	else if( strcmp( argv[ 1 ], "cpu1" ) == 0 )
	{
		s = new CpuSolver1();
	}
	else if( strcmp( argv[ 1 ], "cpu2" ) == 0 )
	{
		s = new CpuSolver2();
	}
	else if( strcmp( argv[ 1 ], "gpu2" ) == 0 )
	{
		s = new Solver2();
	}
	else if( strcmp( argv[ 1 ], "gpu6" ) == 0 )
	{
		s = new Solver6();
	}
	else if( strcmp( argv[ 1 ], "gpu7" ) == 0 )
	{
		s = new Solver7();
	}
	else if( strcmp( argv[ 1 ], "gpu9" ) == 0 )
	{
		s = new Solver9();
	}
	else if( strcmp( argv[ 1 ], "gpu10" ) == 0 )
	{
		s = new Solver10();
	}
	else if( strcmp( argv[ 1 ], "gpu11" ) == 0 )
	{
		s = new Solver11();
	}
	else
	{
		std::cout << "Invalid solver name." << std::endl;
		return EXIT_FAILURE;
	}

	if( !inputUInt( argv[ 2 ], N ) )
	{
		std::cout << "Invalid value for: N" << std::endl;
		delete s;
		return EXIT_FAILURE;
	}
	if( !inputUInt( argv[ 3 ], n ) )
	{
		std::cout << "Invalid value for: n" << std::endl;
		delete s;
		return EXIT_FAILURE;
	}
	t = atoi( argv[ 4 ] );

	s->np = ( 1 << N );
	s->ns = ( 1 << n );
	s->L = L;
	s->T = T;
	s->u0 = u0;
	s->u1 = u1;
	s->sol = sol;
	s->threads = ( 1 << t );
	
	printParams( *s );

	auto start = std::chrono::high_resolution_clock::now();
	s->solve();
	auto elapsed = std::chrono::high_resolution_clock::now() - start;
	double realTime = (double)elapsed.count()
		* std::chrono::high_resolution_clock::period::num
		/ std::chrono::high_resolution_clock::period::den;

	std::cout << "Results:" << std::endl;
	std::cout << "\tTime:\t\t" << realTime << 's' << std::endl;
	std::cout << "\tError:\t\t" << s->error.back() << std::endl;

	char filename[ 256 ];
	sprintf( filename, "%s-N%i-n%i-t%i.txt", s->getName(), N, n, t );

	std::ofstream file( filename );
	file << "Solver N n L T t Error Time" << std::endl;
	file << s->getName() << ' ' << N << ' ' << n << ' ' << L << ' ' << T
		<< ' ' << t << ' ' << s->error.back() << ' ' << realTime << std::endl;
	file.close();

	delete s;

	return 0;
}
