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

#ifndef ENGINE_CUDA_H
#define ENGINE_CUDA_H

#ifdef CUDA_SUPPORT

#include "engine.h"

class Operator;

/**
 * GPU-accelerated FDTD engine using CUDA.
 *
 * Stores field and coefficient arrays on GPU device memory.
 * Core E/H update loops run as CUDA kernels. Extensions fall back
 * to CPU with automatic GPU<->CPU synchronization.
 *
 * Memory layout matches ArrayNIJK (N-I-J-K ordering) with Z innermost,
 * giving coalesced GPU memory access for adjacent threads.
 */
class Engine_CUDA : public Engine
{
public:
	static Engine_CUDA* New(const Operator* op);
	virtual ~Engine_CUDA();

	virtual void Init();
	virtual void Reset();

	virtual bool IterateTS(unsigned int iterTS);

	// Host-side field access (triggers GPU<->CPU transfer for single elements)
	virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const;
	virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const;
	virtual void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value);
	virtual void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value);

	// Bulk sync methods for extension CPU fallback
	void SyncVoltToHost();
	void SyncVoltToDevice();
	void SyncCurrToHost();
	void SyncCurrToDevice();

protected:
	Engine_CUDA(const Operator* op);

	// Device memory pointers (GPU)
	float* d_volt;   // [3 * Nx * Ny * Nz]
	float* d_curr;
	float* d_vv;     // operator coefficients (read-only on GPU)
	float* d_vi;
	float* d_ii;
	float* d_iv;

	size_t m_totalSize;  // 3 * Nx * Ny * Nz (total floats per array)
	size_t m_totalBytes; // m_totalSize * sizeof(float)

	bool m_hasExtensions;
};

#endif // CUDA_SUPPORT
#endif // ENGINE_CUDA_H
