#pragma once

#include <vector>
#include "spdiamat.h"
#include "solverbase.h"

class Solver10 : public SolverBase
{
public:
	Solver10();
	~Solver10();

	void solve();

	virtual const char* getName() const;
};
