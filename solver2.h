#pragma once

#include "solverbase.h"
#include <vector>
#include "spdiamat.h"

class Solver2 : public SolverBase
{
public:
	Solver2();
	~Solver2();

	void solve();

	virtual const char* getName() const;

private:
	static SpDiaMat gpuAllocFDMatrixInner( real l2, unsigned int ip );
	static SpDiaMat gpuAllocFDMatrixLeft( real l2, unsigned int ip );
	static SpDiaMat gpuAllocFDMatrixRight( real l2, unsigned int ip );
	static void gpuFreeFDMatrix( SpDiaMat& mat );
};
