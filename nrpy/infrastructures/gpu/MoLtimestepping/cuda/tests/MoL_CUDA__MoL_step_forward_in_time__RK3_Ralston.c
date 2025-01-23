#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * GPU Kernel: rk_substep_1_gpu.
 * GPU Kernel to compute RK substep 1.
 */
__global__ static void rk_substep_1_gpu(const size_t streamid, REAL *restrict k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs,
                                        REAL *restrict y_n_gfs, const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params[streamid].Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params[streamid].Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params[streamid].Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i = tid0; i < Ntot; i += stride0) {
    const REAL k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfsL = k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    static constexpr REAL RK_Rational_1_2 = 1.0 / 2.0;
    k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs[i] = RK_Rational_1_2 * dt * k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfsL + y_n_gfsL;
  }
}

/**
 * Runge-Kutta function for substep 1.
 */
static void rk_substep_1(params_struct *restrict params, REAL *restrict k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs, REAL *restrict y_n_gfs,
                         const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  const size_t threads_in_x_dir = 32;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Ntot + threads_in_x_dir - 1) / threads_in_x_dir, 1, 1);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  rk_substep_1_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(streamid, k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs,
                                                                                  y_n_gfs, dt);
  cudaCheckErrors(cudaKernel, "rk_substep_1_gpu failure");
}

/**
 * GPU Kernel: rk_substep_2_gpu.
 * GPU Kernel to compute RK substep 2.
 */
__global__ static void rk_substep_2_gpu(const size_t streamid, REAL *restrict k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs,
                                        REAL *restrict k2_or_y_nplus_a32_k2_gfs, REAL *restrict y_n_gfs, const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params[streamid].Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params[streamid].Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params[streamid].Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i = tid0; i < Ntot; i += stride0) {
    const REAL k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfsL = k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs[i];
    const REAL k2_or_y_nplus_a32_k2_gfsL = k2_or_y_nplus_a32_k2_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    static constexpr REAL RK_Rational_4_9 = 4.0 / 9.0;
    static constexpr REAL RK_Rational_5_9 = 5.0 / 9.0;
    static constexpr REAL RK_Rational_1_3 = 1.0 / 3.0;
    static constexpr REAL RK_Rational_3_4 = 3.0 / 4.0;
    k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs[i] = RK_Rational_1_3 * dt * k2_or_y_nplus_a32_k2_gfsL +
                                                            RK_Rational_4_9 * k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfsL +
                                                            RK_Rational_5_9 * y_n_gfsL;
    k2_or_y_nplus_a32_k2_gfs[i] = RK_Rational_3_4 * dt * k2_or_y_nplus_a32_k2_gfsL + y_n_gfsL;
  }
}

/**
 * Runge-Kutta function for substep 2.
 */
static void rk_substep_2(params_struct *restrict params, REAL *restrict k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs,
                         REAL *restrict k2_or_y_nplus_a32_k2_gfs, REAL *restrict y_n_gfs, const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  const size_t threads_in_x_dir = 32;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Ntot + threads_in_x_dir - 1) / threads_in_x_dir, 1, 1);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  rk_substep_2_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(streamid, k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs,
                                                                                  k2_or_y_nplus_a32_k2_gfs, y_n_gfs, dt);
  cudaCheckErrors(cudaKernel, "rk_substep_2_gpu failure");
}

/**
 * GPU Kernel: rk_substep_3_gpu.
 * GPU Kernel to compute RK substep 3.
 */
__global__ static void rk_substep_3_gpu(const size_t streamid, REAL *restrict k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs,
                                        REAL *restrict y_n_gfs, const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params[streamid].Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params[streamid].Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params[streamid].Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i = tid0; i < Ntot; i += stride0) {
    const REAL k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfsL = k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    static constexpr REAL RK_Rational_4_9 = 4.0 / 9.0;
    y_n_gfs[i] = RK_Rational_4_9 * dt * y_n_gfsL + k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfsL;
  }
}

/**
 * Runge-Kutta function for substep 3.
 */
static void rk_substep_3(params_struct *restrict params, REAL *restrict k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs, REAL *restrict y_n_gfs,
                         const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  const size_t threads_in_x_dir = 32;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Ntot + threads_in_x_dir - 1) / threads_in_x_dir, 1, 1);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  rk_substep_3_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(streamid, k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs,
                                                                                  y_n_gfs, dt);
  cudaCheckErrors(cudaKernel, "rk_substep_3_gpu failure");
}

/**
 * Method of Lines (MoL) for "RK3 Ralston" method: Step forward one full timestep.
 *
 */
void MoL_step_forward_in_time(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  // C code implementation of -={ RK3 Ralston }=- Method of Lines timestepping.

  // First set the initial time:
  const REAL time_start = commondata->time;

  // In a diagonal RK3 method like this one, only 3 gridfunctions need be defined. Below implements this approach.
  // Using y_n_gfs as input, k1 and apply boundary conditions
  // -={ START k3 substep }=-
  // RHS evaluation:
  //    1. Add k3 to the running total and save to y_n
  // Post-RHS evaluation:
  //    1. Apply post-RHS to y_n
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 7.50000000000000000e-01 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    MAYBE_UNUSED REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    MAYBE_UNUSED REAL *restrict k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs =
        griddata[grid].gridfuncs.k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs;
    MAYBE_UNUSED REAL *restrict k2_or_y_nplus_a32_k2_gfs = griddata[grid].gridfuncs.k2_or_y_nplus_a32_k2_gfs;
    MAYBE_UNUSED REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    MAYBE_UNUSED params_struct *restrict params = &griddata[grid].params;
    MAYBE_UNUSED REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++)
      xx[ww] = griddata[grid].xx[ww];
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, k2_or_y_nplus_a32_k2_gfs, y_n_gfs);
    rk_substep_3(params, k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs, y_n_gfs, commondata->dt);
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, y_n_gfs);
  }
  // -={ END k3 substep }=-

  // Adding dt to commondata->time many times will induce roundoff error,
  //   so here we set time based on the iteration number.
  commondata->time = (REAL)(commondata->nn + 1) * commondata->dt;

  // Finally, increment the timestep n:
  commondata->nn++;
}
