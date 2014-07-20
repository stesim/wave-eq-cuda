#pragma once

#include <vector>

class SolverBase
{
public:
	virtual ~SolverBase() { }

	virtual void solve() = 0;
	virtual const char* getName() const = 0;

public:
	unsigned int threads;
	unsigned int np;
	unsigned int ns;
	double L;
	double T;
	double (*u0)( double );
	double (*u1)( double );
	double (*sol)( double, double );
	double nc;
	double dc;

	std::vector<double> solution;
	std::vector<double> error;
};
