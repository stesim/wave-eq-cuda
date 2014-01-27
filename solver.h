#pragma once

#include <vector>

class Solver
{
public:
	Solver();
	~Solver();

	void solve();

private:
	static void generateFDMatrix(
			double l2,
			unsigned int ip,
			std::vector<int>& offsets,
			std::vector<double>& values );

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
};
