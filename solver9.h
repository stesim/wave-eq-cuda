#pragma once

#include <vector>
#include "spdiamat.h"
#include "solverbase.h"

class Solver9 : public SolverBase
{
public:
	Solver9();
	~Solver9();

	void solve();

	virtual const char* getName() const;

private:
	static SpDiaMat gpuAllocFDMatrixInner( double l2, unsigned int ip, unsigned int offset );
	static SpDiaMat gpuAllocFDMatrixLeft( double l2, unsigned int ip, unsigned int offset );
	static SpDiaMat gpuAllocFDMatrixRight( double l2, unsigned int ip, unsigned int offset );
};
