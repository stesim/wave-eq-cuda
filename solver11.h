#pragma once

#include <vector>
#include "spdiamat.h"
#include "solverbase.h"

class Solver11 : public SolverBase
{
public:
	Solver11();
	~Solver11();

	void solve();

	virtual const char* getName() const;

private:
	inline void calculateError( real* z, unsigned int k,
			unsigned int ip, unsigned int nsteps, real dt, real h, real* x );
};
