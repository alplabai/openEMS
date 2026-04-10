/*
*	Copyright (C) 2025-2026 alpLab (alplabai)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#ifndef ENGINE_EXT_UPML_CUDA_H
#define ENGINE_EXT_UPML_CUDA_H

#ifdef CUDA_SUPPORT

#include "engine_ext_upml.h"

class Engine_CUDA;

/**
 * CUDA-accelerated UPML extension. Runs PML flux updates entirely on GPU,
 * eliminating the GPU<->CPU sync bottleneck for PML boundaries.
 *
 * PML coefficient arrays (vv, vvfo, vvfn, ii, iifo, iifn) and flux state
 * arrays (volt_flux, curr_flux) are stored in GPU device memory.
 * The extension directly reads/writes the Engine_CUDA device field arrays.
 */
class Engine_Ext_UPML_CUDA : public Engine_Ext_UPML
{
public:
	Engine_Ext_UPML_CUDA(Operator_Ext_UPML* op_ext);
	virtual ~Engine_Ext_UPML_CUDA();

	virtual void DoPreVoltageUpdates() override;
	virtual void DoPreVoltageUpdates(int) override { DoPreVoltageUpdates(); }
	virtual void DoPostVoltageUpdates() override;
	virtual void DoPostVoltageUpdates(int) override { DoPostVoltageUpdates(); }
	virtual void DoPreCurrentUpdates() override;
	virtual void DoPreCurrentUpdates(int) override { DoPreCurrentUpdates(); }
	virtual void DoPostCurrentUpdates() override;
	virtual void DoPostCurrentUpdates(int) override { DoPostCurrentUpdates(); }

	bool IsGPUExtension() const { return true; }

protected:
	void InitCUDA();
	void FreeCUDA();

	Engine_CUDA* m_Eng_CUDA;

	// Device memory for PML coefficients (read-only on GPU)
	float* d_vv;
	float* d_vvfo;
	float* d_vvfn;
	float* d_ii;
	float* d_iifo;
	float* d_iifn;

	// Device memory for PML flux state
	float* d_volt_flux;
	float* d_curr_flux;

	// PML region dimensions
	unsigned int m_pml_Nx, m_pml_Ny, m_pml_Nz;
	unsigned int m_startX, m_startY, m_startZ;
	// Global domain dimensions (for indexing into engine arrays)
	unsigned int m_global_Nx, m_global_Ny, m_global_Nz;
	size_t m_pml_totalSize;  // 3 * Nx * Ny * Nz
};

#endif // CUDA_SUPPORT
#endif // ENGINE_EXT_UPML_CUDA_H
