#include <iostream>
#include "solver.h"
#include "solver2.h"
#include "solver3.h"
#include "solver4.h"
#include "solver5.h"
#include "solver6.h"
#include "solver7.h"
#include "solver8.h"
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
	Solver6 s;
	s.np = ( 1 << 14 );
	s.ns = ( 1 << 7 );
	s.L = 150.0;
	s.T = 100.0;
	s.u0 = u0;
	s.u1 = u1;
	s.sol = sol;
	//s.threads = 1024;

	for( s.threads = 32; s.threads <= 128; s.threads *= 2 )
	{
		std::cout << "threads: " << s.threads << std::endl;

		auto start = std::chrono::high_resolution_clock::now();
		s.solve();
		auto elapsed = std::chrono::high_resolution_clock::now() - start;
		std::cout << "time: " << (double)elapsed.count() * std::chrono::high_resolution_clock::period::num / std::chrono::high_resolution_clock::period::den << 's' << std::endl;
		std::cout << std::endl;
	}

	std::cout << "error (k=0):    " << s.error[ 0 ] << std::endl;
	std::cout << "error (k=kmax): " << s.error[ s.error.size() - 1 ] << std::endl;

	/*
	std::ofstream file;
	file.open( "error.txt" );
	for( unsigned int i = 0; i < s.error.size(); ++i )
	{
		file << s.error[ i ] << std::endl;
	}
	file.close();

	file.open( "solution.txt" );
	for( unsigned int i = 0; i < s.solution.size(); ++i )
	{
		file << s.solution[ i ] << std::endl;
	}
	file.close();
	*/

	return 0;
}
