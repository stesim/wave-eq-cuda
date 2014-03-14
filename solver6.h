#pragma once

#include <vector>
#include "spdiamat.h"
#include "solverbase.h"

class Solver6 : public SolverBase
{
public:
	Solver6();
	~Solver6();

	void solve();

	virtual const char* getName() const;

private:
	static SpDiaMat gpuAllocFDMatrixInner( double l2, unsigned int ip );
	static SpDiaMat gpuAllocFDMatrixLeft( double l2, unsigned int ip );
	static SpDiaMat gpuAllocFDMatrixRight( double l2, unsigned int ip );
	static void gpuFreeFDMatrix( SpDiaMat& mat );
};
