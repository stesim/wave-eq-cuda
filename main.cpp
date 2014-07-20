#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <cstring>

#include "solverbase.h"
#include "cpusolver0.h"
#include "cpusolver1.h"
#include "cpusolver2.h"
#include "cpusolver3.h"
#include "solver7.h"
#include "solver11.h"

// initial value
double u0( double x )
{
	return 1.0 / ( 1.0 + x * x );
}

// derivative initial value
double u1( double x )
{
	return 0.0;
}

// analytical solution
double sol( double x, double t )
{
	double a = x - t;
	double b = x + t;
	return ( 1.0 / ( 1.0 + a * a ) + 1.0 / ( 1.0 + b * b ) ) / 2.0;
}

// parse unsigned integer from string
inline bool inputUInt( const char* str, unsigned int& val )
{
	val = atoi( str );
	return ( val != 0 );
}

inline bool inputDouble( const char* str, double& val )
{
	char* endptr = NULL;
	val = strtod( str, &endptr );
	return ( *endptr == 0 );
}

// print parameter overview
inline void printParams( const SolverBase& solver )
{
	std::cout << "Parameter overview:" << std::endl;
	std::cout << "\tL:\t\t" << solver.L << std::endl;
	std::cout << "\tT:\t\t" << solver.T << std::endl;
	std::cout << "\tnp:\t\t" << solver.np << std::endl;
	std::cout << "\tns:\t\t" << solver.ns << std::endl;
	std::cout << "\tdc:\t\t" << solver.dc << std::endl;
	std::cout << "\tnc:\t\t" << solver.nc << std::endl;
	std::cout << "\tSolver:\t\t" << solver.getName() << std::endl;
	std::cout << "\tThreads:\t" << solver.threads << std::endl;
}

// parse single double argument
void parseArgDouble( int argc, char* argv[], int& idx, double& val, bool& valid )
{
	if( ++idx >= argc || !inputDouble( argv[ idx ], val ) )
	{
		valid = false;
		std::cout << "Invalid input for option " <<
			argv[ idx - 1 ] << std::endl;
	}
}

// parse single unsigned int argument
void parseArgUInt( int argc, char* argv[], int& idx,
		unsigned int& val, bool& valid )
{
	if( ++idx >= argc || !inputUInt( argv[ idx ], val ) )
	{
		valid = false;
		std::cout << "Invalid input for option " <<
			argv[ idx - 1 ] << std::endl;
	}
}

// parse input arguments
bool parseArgs( int argc, char* argv[],
		double& L, double& T, double& dc, double& nc,
		unsigned int& N, unsigned int& n, unsigned int& t, const char*& solver )
{
	bool retVal = true;
	for( int i = 1; i < argc; ++i )
	{
		if( strcmp( argv[ i ], "-L" ) == 0 )
		{
			parseArgDouble( argc, argv, i, L, retVal );
		}
		else if( strcmp( argv[ i ], "-T" ) == 0 )
		{
			parseArgDouble( argc, argv, i, T, retVal );
		}
		else if( strcmp( argv[ i ], "-dc" ) == 0 )
		{
			parseArgDouble( argc, argv, i, dc, retVal );
		}
		else if( strcmp( argv[ i ], "-nc" ) == 0 )
		{
			parseArgDouble( argc, argv, i, nc, retVal );
		}
		else if( strcmp( argv[ i ], "-N" ) == 0 )
		{
			parseArgUInt( argc, argv, i, N, retVal );
		}
		else if( strcmp( argv[ i ], "-n" ) == 0 )
		{
			parseArgUInt( argc, argv, i, n, retVal );
		}
		else if( strcmp( argv[ i ], "-t" ) == 0 )
		{
			parseArgUInt( argc, argv, i, t, retVal );
		}
		else if( strcmp( argv[ i ], "-solver" ) == 0 )
		{
			if( i + 1 >= argc )
			{
				solver = NULL;
			}
			else
			{
				solver = argv[ i + 1 ];
				++i;
			}
		}
		else if( strcmp( argv[ i ], "--help" ) == 0 ||
				strcmp( argv[ i ], "-help" ) == 0 )
		{
			std::cout << "Solves non-linear wave equation: u_{tt} = u_{xx} - d_c u + n_c u^3" << std::endl
				<< std::endl
				<< "Available parameters:" << std::endl
				<< "\t-L        Size of the domain." << std::endl
				<< "\t-T        Target time for which the solution is required." << std::endl
				<< "\t-dc       Non-linear coefficient in the wave equation." << std::endl
				<< "\t-N        Exponent of the number of nodes used for discretization." << std::endl
				<< "\t-n        Exponent of the number of subdomains used." << std::endl
				<< "\t-t        Exponent of number of threads used for parallel CPU solvers." << std::endl
				<< "\t          Exponent of number of threads per block for vectorized GPU solver." << std::endl
				<< "\t-solver   Solver type to be used for computation. Possible values:" << std::endl
				<< "\t             cpu0:    Serial vectorized CPU solver." << std::endl
				<< "\t             cpu1:    Parallel vectorized CPU solver." << std::endl
				<< "\t             cpu2:    Parallel scalar CPU solver." << std::endl
				<< "\t             cpu3:    Serial scalar CPU solver." << std::endl
				<< "\t             gpu7:    Vectorized GPU solver." << std::endl
				<< "\t             gpu11:   Scalar GPU solver. (Default)" << std::endl;
			retVal = false;
		}
		else
		{
			std::cout << "Invalid parameter: " << argv[ i ] << std::endl;
			retVal = false;
		}
	}
	return retVal;
}

// save results to files
inline void saveResults( const SolverBase& s, unsigned int N,
		unsigned int n, unsigned int t, double time )
{
	char filenameBase[ 128 ];
	sprintf( filenameBase, "%s-N%i-n%i-t%i-L%f-T%f-dc%f-nc%f", s.getName(),
			N, n, t, s.L, s.T, s.dc, s.nc );

	char filename[ 256 ];
	sprintf( filename, "%s-details.txt", filenameBase );

	std::ofstream file( filename );
	file << "solver  " << s.getName() << std::endl;
	file << "L       " << s.L << std::endl;
	file << "T       " << s.T << std::endl;
	file << "dc      " << s.dc << std::endl;
	file << "nc      " << s.nc << std::endl;
	file << "np      " << s.np << std::endl;
	file << "ns      " << s.ns << std::endl;
	file << "threads " << s.threads << std::endl;
	file << "time    " << time << std::endl;
	file << "error   " << s.error.back() << std::endl;
	file.close();

	sprintf( filename, "%s-solution.txt", filenameBase );

	file.open( filename );
	double h = 2 * s.L / ( s.np - 1 );
	// save solution.
	//    first column: x axis value
	//    second column: solution value
	for( unsigned int i = 0; i < s.np; ++i )
	{
		double x = i * h - s.L;
		file << x << "\t" << s.solution[ i ] << std::endl;
	}
	file.close();
}

int main( int argc, char* argv[] )
{
	// default parameters
	double L = 150.0;
	double T = 100.0;
	double dc = 0.0;
	double nc = 0.0;
	
	unsigned int N = 21;
	unsigned int n = 11;
	unsigned int t = 3;

	static const char* defaultSolver = "gpu11";

	const char* solverName = defaultSolver;

	// parse input parameters
	if( !parseArgs( argc, argv, L, T, dc, nc, N, n, t, solverName ) )
	{
		return EXIT_FAILURE;
	}

	SolverBase* s;
	if( strcmp( solverName, "cpu0" ) == 0 )
	{
		s = new CpuSolver0();
	}
	else if( strcmp( solverName, "cpu1" ) == 0 )
	{
		s = new CpuSolver1();
	}
	else if( strcmp( solverName, "cpu2" ) == 0 )
	{
		s = new CpuSolver2();
	}
	else if( strcmp( solverName, "cpu3" ) == 0 )
	{
		s = new CpuSolver3();
	}
	else if( strcmp( solverName, "gpu7" ) == 0 )
	{
		s = new Solver7();
	}
	else if( strcmp( solverName, "gpu11" ) == 0 )
	{
		s = new Solver11();
	}
	else
	{
		std::cout << "Invalid solver name." << std::endl;
		return EXIT_FAILURE;
	}

	s->np = ( 1 << N );
	s->ns = ( 1 << n );
	s->L = L;
	s->T = T;
	//s->T = static_cast<double>( 1 << 20 );
	s->dc = dc;
	s->nc = nc;
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

	saveResults( *s, N, n, t, realTime );

	delete s;

	return 0;
}
