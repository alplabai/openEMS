/*
*	Copyright (C) 2025 alpLab (alplabai)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "engine_cuda.h"

#ifdef CUDA_SUPPORT

#include <cuda_runtime.h>
#include <iostream>
#include "engine_cuda_kernels.h"
#include "tools/openems_error.h"

using std::cerr;
using std::endl;

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) { \
		cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
		     << cudaGetErrorString(err) << endl; \
		throw openEMS_InternalError(std::string("CUDA error: ") + cudaGetErrorString(err)); \
	} \
} while(0)

Engine_CUDA::Engine_CUDA(const Operator* op) : Engine(op)
{
	m_type = Engine::CUDA;
	d_volt = nullptr;
	d_curr = nullptr;
	d_vv = nullptr;
	d_vi = nullptr;
	d_ii = nullptr;
	d_iv = nullptr;
	m_totalSize = 0;
	m_totalBytes = 0;
	m_hasExtensions = false;
}

Engine_CUDA* Engine_CUDA::New(const Operator* op)
{
	cout << "Create CUDA engine..." << endl;

	// Check for CUDA device
	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	if (err != cudaSuccess || deviceCount == 0)
		throw openEMS_SetupError("Engine_CUDA: No CUDA-capable GPU found");

	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	cout << "  GPU: " << prop.name << ", "
	     << (prop.totalGlobalMem / (1024*1024)) << " MB, "
	     << "Compute " << prop.major << "." << prop.minor << endl;

	Engine_CUDA* e = new Engine_CUDA(op);
	e->Init();
	return e;
}

Engine_CUDA::~Engine_CUDA()
{
	Reset();
}

void Engine_CUDA::Init()
{
	// Initialize base engine (allocates host arrays)
	Engine::Init();

	m_totalSize = 3 * (size_t)numLines[0] * numLines[1] * numLines[2];
	m_totalBytes = m_totalSize * sizeof(float);

	cout << "  CUDA: Allocating " << (m_totalBytes * 8 / (1024*1024))
	     << " MB GPU memory (8 arrays)" << endl;

	// Allocate device arrays
	CUDA_CHECK(cudaMalloc(&d_volt, m_totalBytes));
	CUDA_CHECK(cudaMalloc(&d_curr, m_totalBytes));
	CUDA_CHECK(cudaMalloc(&d_vv,   m_totalBytes));
	CUDA_CHECK(cudaMalloc(&d_vi,   m_totalBytes));
	CUDA_CHECK(cudaMalloc(&d_ii,   m_totalBytes));
	CUDA_CHECK(cudaMalloc(&d_iv,   m_totalBytes));

	// Upload field arrays (initialized to zero by Engine::Init)
	CUDA_CHECK(cudaMemcpy(d_volt, volt_ptr->data(), m_totalBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_curr, curr_ptr->data(), m_totalBytes, cudaMemcpyHostToDevice));

	// Upload operator coefficients (read-only on GPU)
	CUDA_CHECK(cudaMemcpy(d_vv, Op->vv_ptr->data(), m_totalBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_vi, Op->vi_ptr->data(), m_totalBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_ii, Op->ii_ptr->data(), m_totalBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_iv, Op->iv_ptr->data(), m_totalBytes, cudaMemcpyHostToDevice));

	m_hasExtensions = (m_Eng_exts.size() > 0);
}

void Engine_CUDA::Reset()
{
	if (d_volt) { cudaFree(d_volt); d_volt = nullptr; }
	if (d_curr) { cudaFree(d_curr); d_curr = nullptr; }
	if (d_vv)   { cudaFree(d_vv);   d_vv = nullptr; }
	if (d_vi)   { cudaFree(d_vi);   d_vi = nullptr; }
	if (d_ii)   { cudaFree(d_ii);   d_ii = nullptr; }
	if (d_iv)   { cudaFree(d_iv);   d_iv = nullptr; }
	m_totalSize = 0;
	m_totalBytes = 0;

	Engine::Reset();
}

bool Engine_CUDA::IterateTS(unsigned int iterTS)
{
	for (unsigned int iter = 0; iter < iterTS; ++iter)
	{
		// --- Voltage update cycle ---
		if (m_hasExtensions)
		{
			SyncVoltToHost();
			SyncCurrToHost();
			DoPreVoltageUpdates();
			SyncVoltToDevice();
		}

		LaunchUpdateVoltages(d_volt, d_curr, d_vv, d_vi,
		                     numLines[0], numLines[1], numLines[2]);

		if (m_hasExtensions)
		{
			cudaDeviceSynchronize();
			SyncVoltToHost();
			DoPostVoltageUpdates();
			Apply2Voltages();
			SyncVoltToDevice();
		}

		// --- Current update cycle ---
		if (m_hasExtensions)
		{
			SyncVoltToHost();
			SyncCurrToHost();
			DoPreCurrentUpdates();
			SyncCurrToDevice();
			SyncVoltToDevice();
		}

		LaunchUpdateCurrents(d_curr, d_volt, d_ii, d_iv,
		                     numLines[0], numLines[1], numLines[2]);

		if (m_hasExtensions)
		{
			cudaDeviceSynchronize();
			SyncCurrToHost();
			DoPostCurrentUpdates();
			Apply2Current();
			SyncCurrToDevice();
		}

		++numTS;
	}

	// Final sync so host arrays are up to date for processing/output
	cudaDeviceSynchronize();

	return true;
}

// --- Bulk sync methods ---

void Engine_CUDA::SyncVoltToHost()
{
	CUDA_CHECK(cudaMemcpy(volt_ptr->data(), d_volt, m_totalBytes, cudaMemcpyDeviceToHost));
}

void Engine_CUDA::SyncVoltToDevice()
{
	CUDA_CHECK(cudaMemcpy(d_volt, volt_ptr->data(), m_totalBytes, cudaMemcpyHostToDevice));
}

void Engine_CUDA::SyncCurrToHost()
{
	CUDA_CHECK(cudaMemcpy(curr_ptr->data(), d_curr, m_totalBytes, cudaMemcpyDeviceToHost));
}

void Engine_CUDA::SyncCurrToDevice()
{
	CUDA_CHECK(cudaMemcpy(d_curr, curr_ptr->data(), m_totalBytes, cudaMemcpyHostToDevice));
}

// --- Single-element access (slow, for CPU extension fallback) ---

FDTD_FLOAT Engine_CUDA::GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	size_t offset = n * numLines[0] * numLines[1] * numLines[2]
	              + x * numLines[1] * numLines[2]
	              + y * numLines[2]
	              + z;
	float val;
	cudaMemcpy(&val, d_volt + offset, sizeof(float), cudaMemcpyDeviceToHost);
	return val;
}

FDTD_FLOAT Engine_CUDA::GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	size_t offset = n * numLines[0] * numLines[1] * numLines[2]
	              + x * numLines[1] * numLines[2]
	              + y * numLines[2]
	              + z;
	float val;
	cudaMemcpy(&val, d_curr + offset, sizeof(float), cudaMemcpyDeviceToHost);
	return val;
}

void Engine_CUDA::SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
{
	size_t offset = n * numLines[0] * numLines[1] * numLines[2]
	              + x * numLines[1] * numLines[2]
	              + y * numLines[2]
	              + z;
	float val = value;
	cudaMemcpy(d_volt + offset, &val, sizeof(float), cudaMemcpyHostToDevice);
}

void Engine_CUDA::SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
{
	size_t offset = n * numLines[0] * numLines[1] * numLines[2]
	              + x * numLines[1] * numLines[2]
	              + y * numLines[2]
	              + z;
	float val = value;
	cudaMemcpy(d_curr + offset, &val, sizeof(float), cudaMemcpyHostToDevice);
}

#endif // CUDA_SUPPORT
