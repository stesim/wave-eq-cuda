#pragma once

#include <vector>
#include "spdiamat.h"

class Solver2
{
public:
	Solver2();
	~Solver2();

	void solve();

private:
	static SpDiaMat gpuAllocFDMatrixInner( double l2, unsigned int ip );
	static SpDiaMat gpuAllocFDMatrixLeft( double l2, unsigned int ip );
	static SpDiaMat gpuAllocFDMatrixRight( double l2, unsigned int ip );
	static void gpuFreeFDMatrix( SpDiaMat& mat );

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
