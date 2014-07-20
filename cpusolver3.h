#pragma once

#include "solverbase.h"
#include <vector>
#include "spdiamat.h"

class CpuSolver3 : public SolverBase
{
public:
	CpuSolver3();
	virtual ~CpuSolver3();

	virtual void solve();

	virtual const char* getName() const;

protected:
	static void calculateFirstStep(
			unsigned int ip,
			real* z,
			const real* f,
			const real* g,
			real dt,
			real l2,
			real a,
			real b );

	static void calculateNSteps(
			unsigned int ip,
			real*& z,
			real*& w,
			real*& u,
			real l2,
			real a,
			real b,
			unsigned int nsteps );
};
