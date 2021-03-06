#pragma once

#include "solverbase.h"
#include <vector>
#include "spdiamat.h"

class CpuSolver0 : public SolverBase
{
public:
	CpuSolver0();
	virtual ~CpuSolver0();

	virtual void solve();

	virtual const char* getName() const;

protected:
	static void calculateFirstStep(
			unsigned int ip,
			real* z,
			const real* f,
			const real* g,
			const SpDiaMat& mat,
			real dt,
			real a,
			real b );

	static void calculateNSteps(
			unsigned int ip,
			real*& z,
			real*& w,
			real*& u,
			const SpDiaMat& mat,
			real a,
			real b,
			unsigned int nsteps );

private:
	static SpDiaMat allocFDMatrix( real l2, unsigned int ip );
};
