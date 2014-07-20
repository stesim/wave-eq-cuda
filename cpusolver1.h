#pragma once

#include "cpusolver0.h"
#include <vector>
#include "spdiamat.h"
#include <thread>
#include <mutex>

class CpuSolver1 : public CpuSolver0
{
public:
	CpuSolver1();
	virtual ~CpuSolver1();

	virtual void solve();

	virtual const char* getName() const;

private:
	static SpDiaMat allocFDMatrixInner( real l2, unsigned int ip );
	static SpDiaMat allocFDMatrixLeft( real l2, unsigned int ip );
	static SpDiaMat allocFDMatrixRight( real l2, unsigned int ip );

	void threadLoop(
			bool firstStep,
			unsigned int ndom,
			unsigned int ip,
			std::vector<real*>& pz,
			std::vector<real*>& pw,
			std::vector<real*>& pu,
			const SpDiaMat& matInner,
			const SpDiaMat& matLeft,
			const SpDiaMat& matRight,
			real dt,
			real a,
			real b,
			unsigned int nsteps );

	void runThreadPool(
			bool firstStep,
			unsigned int ndom,
			unsigned int ip,
			std::vector<real*>& pz,
			std::vector<real*>& pw,
			std::vector<real*>& pu,
			const SpDiaMat& matInner,
			const SpDiaMat& matLeft,
			const SpDiaMat& matRight,
			real dt,
			real a,
			real b,
			unsigned int nsteps );

private:
	thread_local static unsigned int threadId;

	unsigned int m_uiThreadId;
	std::mutex m_Mutex;
};
