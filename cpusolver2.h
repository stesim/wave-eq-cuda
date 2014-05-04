#pragma once

#include "solverbase.h"
#include <vector>
#include "spdiamat.h"
#include <thread>
#include <mutex>

class CpuSolver2 : public SolverBase
{
public:
	CpuSolver2();
	virtual ~CpuSolver2();

	virtual void solve();

	virtual const char* getName() const;

private:
	void threadLoop(
			bool firstStep,
			unsigned int ndom,
			unsigned int ip,
			std::vector<real*>& pz,
			std::vector<real*>& pw,
			std::vector<real*>& pu,
			real dt,
			real l2,
			unsigned int nsteps );

	void runThreadPool(
			bool firstStep,
			unsigned int ndom,
			unsigned int ip,
			std::vector<real*>& pz,
			std::vector<real*>& pw,
			std::vector<real*>& pu,
			real dt,
			real l2,
			unsigned int nsteps );

private:
	thread_local static unsigned int threadId;

	unsigned int m_uiThreadId;
	std::mutex m_Mutex;
};
