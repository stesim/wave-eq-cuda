#pragma once

#include "solverbase.h"
#include <vector>

class Solver : public SolverBase
{
public:
	Solver();
	~Solver();

	void solve();

	virtual const char* getName() const;

private:
	static void generateFDMatrix(
			double l2,
			unsigned int ip,
			std::vector<int>& offsets,
			std::vector<double>& values );
/*
public:
	unsigned int threads;
	unsigned int np;
	unsigned int ns;
	double L;
	double T;
	double (*u0)( double );
	double (*u1)( double );
	double (*sol)( double, double );

	std::vector<double> solution;
	std::vector<double> error;
*/
};
