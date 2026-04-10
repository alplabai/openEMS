/*
*	Copyright (C) 2025-2026 alpLab (alplabai)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#ifndef ENGINE_EXT_EXCITATION_CUDA_H
#define ENGINE_EXT_EXCITATION_CUDA_H

#ifdef CUDA_SUPPORT

#include "engine_ext_excitation.h"

class Engine_CUDA;

/**
 * CUDA-accelerated excitation extension.
 * Uploads excitation arrays (positions, amplitudes, delays, signal waveform)
 * to GPU and runs sparse excitation kernel on device.
 */
class Engine_Ext_Excitation_CUDA : public Engine_Ext_Excitation
{
public:
	Engine_Ext_Excitation_CUDA(Operator_Ext_Excitation* op_ext);
	virtual ~Engine_Ext_Excitation_CUDA();

	virtual void Apply2Voltages() override;
	virtual void Apply2Current() override;

	bool IsGPUExtension() const { return true; }

protected:
	void InitCUDA();
	void FreeCUDA();
	bool m_cuda_init;

	Engine_CUDA* m_Eng_CUDA;

	// Voltage excitation device arrays
	float* d_Volt_amp;
	unsigned int* d_Volt_delay;
	unsigned short* d_Volt_dir;
	unsigned int* d_Volt_index_x;
	unsigned int* d_Volt_index_y;
	unsigned int* d_Volt_index_z;

	// Current excitation device arrays
	float* d_Curr_amp;
	unsigned int* d_Curr_delay;
	unsigned short* d_Curr_dir;
	unsigned int* d_Curr_index_x;
	unsigned int* d_Curr_index_y;
	unsigned int* d_Curr_index_z;

	// Signal waveform
	float* d_exc_volt_signal;
	float* d_exc_curr_signal;
	unsigned int m_signal_length;
	int m_signal_period;
};

#endif // CUDA_SUPPORT
#endif // ENGINE_EXT_EXCITATION_CUDA_H
