# openEMS (alpLab fork)

This is the [alpLab](https://github.com/alplabai) fork of [openEMS](https://github.com/thliebig/openEMS), maintained for integration into [Signex](https://github.com/alplabai/alp-eda) as the primary PCB electromagnetic field solver.

**Upstream**: [https://github.com/thliebig/openEMS](https://github.com/thliebig/openEMS)<br />
**Upstream Website**: [https://openEMS.de](https://openEMS.de)<br />
**Upstream Docs**: [https://docs.openEMS.de](https://docs.openEMS.de)<br />

## Changes from upstream

All changes are kept minimal and upstreamable.

- Replace all `exit()` calls with exception hierarchy (`openEMS_SetupError`, `openEMS_AllocationError`, `openEMS_InternalError`) for safe shared-library use
- Replace `volatile` with `std::atomic` for proper thread safety in multi-threaded engine
- Fix `setlocale` race conditions with RAII `ScopedNumericLocale` guard
- `SnapToMeshLine` O(N) linear search replaced with O(log N) binary search (~14x speedup)
- Fix `diff_pow` and `m_LM_pos[n]` memory leaks, `size_t` overflow
- Merged upstream WIP: multi-threaded SAR (IEEE/IEC compliant), HDF5 improvements, arraylib

## Features

- Fully 3D Cartesian and cylindrical coordinates graded mesh
- Multi-threading, SIMD (SSE) and MPI support for high speed FDTD
- Octave/Matlab and Python interface
- Dispersive material (Drude/Lorentz/Debye type)
- Lumped RLC elements, conducting sheet model
- MSL/Stripline/CPW/Waveguide ports with S-parameter extraction
- Field dumps in time and frequency domain as VTK or HDF5 file format
- Near-to-far-field (NF2FF) transformation
- Multi-threaded SAR calculation with IEEE/IEC compliance

## Building (MSYS2 MinGW on Windows)

```bash
pacman -S mingw-w64-x86_64-{gcc,cmake,boost,hdf5,vtk,tinyxml,cgal,nlohmann-json}

mkdir -p build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_INSTALL_PREFIX=../../install
mingw32-make -j$(nproc) install
```

## License

openEMS is licensed under the terms of the GPLv3, see <http://www.gnu.org/licenses/>.
