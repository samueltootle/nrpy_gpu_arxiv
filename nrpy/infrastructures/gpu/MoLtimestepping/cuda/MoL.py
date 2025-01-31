"""
Module for producing C codes related to MoL timestepping within the BHaH infrastructure.
This includes implementation details and functions for
allocating and deallocating the necessary memory. This modular is
specifically focused on utilizing CUDA parallelization when generating
code

Authors: Brandon Clark
         Zachariah B. Etienne (maintainer) zachetie **at** gmail **dot**
         com Samuel D. Tootle sdtootle **at** gmail **dot** com
"""

import os  # Standard Python module for multiplatform OS-level functions
from typing import Dict, List, Tuple, Union

import sympy as sp  # Import SymPy, a computer algebra system written entirely in Python

import nrpy.c_function as cfc
import nrpy.helpers.gpu.gpu_kernel as gputils
import nrpy.infrastructures.BHaH.BHaH_defines_h as BHaH_defines_overload
import nrpy.params as par  # NRPy+: Parameter interface
from nrpy.helpers.generic import superfast_uniq
from nrpy.infrastructures.BHaH import BHaH_defines_h, griddata_commondata
from nrpy.infrastructures.gpu.MoLtimestepping import base_MoL

# fmt: off
_ = par.CodeParameter("int", __name__, "nn_0", add_to_parfile=False, add_to_set_CodeParameters_h=True, commondata=True)
_ = par.CodeParameter("int", __name__, "nn", add_to_parfile=False, add_to_set_CodeParameters_h=True, commondata=True)
_ = par.CodeParameter("REAL", __name__, "CFL_FACTOR", 0.5, commondata=True)
_ = par.CodeParameter("REAL", __name__, "dt", add_to_parfile=False, add_to_set_CodeParameters_h=True, commondata=True)
_ = par.CodeParameter("REAL", __name__, "t_0", add_to_parfile=False, add_to_set_CodeParameters_h=True, commondata=True)
_ = par.CodeParameter("REAL", __name__, "time", add_to_parfile=False, add_to_set_CodeParameters_h=True, commondata=True)
_ = par.CodeParameter("REAL", __name__, "t_final", 10.0, commondata=True)
# fmt: on

# Update core_modules to use correct key for ordering
for idx, key in enumerate(BHaH_defines_overload.core_modules_list):
    if "nrpy.infrastructures.BHaH.MoLtimestepping" in key:
        BHaH_defines_overload.core_modules_list[idx] = str(__name__)


class register_CFunction_MoL_malloc(base_MoL.base_register_CFunction_MoL_malloc):
    """
    Register MoL_malloc_y_n_gfs() and MoL_malloc_non_y_n_gfs(), allocating memory for the gridfunctions indicated.

    :param Butcher_dict: Dictionary of Butcher tables for the MoL method.
    :param MoL_method: Method for the Method of Lines.
    :param which_gfs: Specifies which gridfunctions to consider ("y_n_gfs" or "non_y_n_gfs").

    :raises ValueError: If the which_gfs parameter is neither "y_n_gfs" nor "non_y_n_gfs".

    Doctest: FIXME
    # >>> register_CFunction_MoL_malloc("Euler", "y_n_gfs")
    """

    def __init__(
        self,
        Butcher_dict: Dict[str, Tuple[List[List[Union[sp.Basic, int, str]]], int]],
        MoL_method: str,
        which_gfs: str,
    ) -> None:

        super().__init__(Butcher_dict, MoL_method, which_gfs)

        # Generate the body of the function

        for gridfunctions in self.gridfunctions_list:
            num_gfs = (
                "NUM_EVOL_GFS" if gridfunctions != "auxevol_gfs" else "NUM_AUXEVOL_GFS"
            )
            # Don't malloc a zero-sized array.
            if num_gfs == "NUM_AUXEVOL_GFS":
                self.body += "  if(NUM_AUXEVOL_GFS > 0) "
            self.body += (
                f"cudaMalloc(&gridfuncs->{gridfunctions}, sizeof(REAL) * {num_gfs} * "
                "Nxx_plus_2NGHOSTS_tot);\n"
                'cudaCheckErrors(malloc, "Malloc failed");\n'
            )

        self.body += f"\ngridfuncs->diagnostic_output_gfs  = gridfuncs->{self.diagnostic_gridfunctions_point_to};\n"
        self.body += f"gridfuncs->diagnostic_output_gfs2 = gridfuncs->{self.diagnostic_gridfunctions2_point_to};\n"

        self.register()


class RKFunction(base_MoL.RKFunction):
    """
    CUDA overload of RKFUNCTION.

    :param fp_type_alias: Infrastructure-specific alias for the C/C++ floating point data type. E.g., 'REAL' or 'CCTK_REAL'.
    :param operator: The operator with respect to which the derivative is taken.
    :param RK_lhs_list: List of LHS expressions for RK substep.
    :param RK_rhs_list: List of RHS expressions for RK substep.
    :param enable_intrinsics: A flag to specify if hardware intrinsics should be used.
    :param cfunc_type: decorators and return type for the RK substep function
    :param rk_step: current step (> 0).  Default (None) assumes Euler step
    :param rational_const_alias: Overload const specifier for Rational definitions
    """

    def __init__(
        self,
        fp_type_alias: str,
        RK_lhs_list: List[sp.Basic],
        RK_rhs_list: List[sp.Basic],
        enable_intrinsics: bool = False,
        cfunc_type: str = "static void",
        rk_step: Union[int, None] = None,
        rational_const_alias: str = "static constexpr",
    ) -> None:
        super().__init__(
            fp_type_alias,
            RK_lhs_list,
            RK_rhs_list,
            enable_intrinsics=enable_intrinsics,
            cfunc_type=cfunc_type,
            rk_step=rk_step,
            rational_const_alias=rational_const_alias,
            intrinsics_str="CUDA",
        )
        self.device_kernel: gputils.GPU_Kernel

    def CFunction_RK_substep_function(self) -> None:
        """Generate a C function based on the given RK substep expression lists."""
        self.body = ""
        self.params: str = "params_struct *restrict params, "
        kernel_body: str = ""
        N_str = ""
        for i in ["0", "1", "2"]:
            kernel_body += f"const int Nxx_plus_2NGHOSTS{i} = d_params[streamid].Nxx_plus_2NGHOSTS{i};\n"
            N_str += f"Nxx_plus_2NGHOSTS{i} *"
        kernel_body += "const int Ntot = Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1*Nxx_plus_2NGHOSTS2*NUM_EVOL_GFS;\n\n"
        kernel_body += (
            "// Kernel thread/stride setup\n"
            "const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;\n"
            "const int stride0 = blockDim.x * gridDim.x;\n\n"
            "for(int i=tid0;i<Ntot;i+=stride0) {\n"
        )

        var_type = "REAL"

        read_list = [
            read
            for el in self.RK_rhs_list
            for read in list(sp.ordered(el.free_symbols))
        ]
        read_list_unique = superfast_uniq(read_list)

        for el in read_list_unique:
            if str(el) != "commondata->dt":
                gfs_el = str(el).replace("gfsL", "gfs[i]")
                self.param_vars.append(gfs_el[:-3])
                self.params += f"{var_type} *restrict {gfs_el[:-3]},"
                kernel_body += f"const {var_type} {el} = {gfs_el};\n"
        for el in self.RK_lhs_list:
            lhs_var = str(el).replace("_gfsL", "_gfs")
            if not lhs_var in self.params:
                self.param_vars.append(lhs_var)
                self.params += f"{var_type} *restrict {lhs_var},"
        self.params += f"const {var_type} dt"

        kernel_params = {k: "REAL *restrict" for k in self.param_vars}
        kernel_params["dt"] = "const REAL"
        self.device_kernel = gputils.GPU_Kernel(
            kernel_body + self.loop_body.replace("commondata->dt", "dt") + "\n}\n",
            kernel_params,
            f"{self.name}_gpu",
            launch_dict={
                "blocks_per_grid": ["(Ntot + threads_in_x_dir - 1) / threads_in_x_dir"],
                "threads_per_block": ["32"],
                "stream": "params->grid_idx % nstreams",
            },
            comments=f"GPU Kernel to compute RK substep {self.rk_step}.",
        )
        prefunc = self.device_kernel.CFunction.full_function

        # Store CFunction
        for j in range(3):
            self.body += (
                f"const int Nxx_plus_2NGHOSTS{j} = params->Nxx_plus_2NGHOSTS{j};\n"
            )
        self.body += "const int Ntot = Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1*Nxx_plus_2NGHOSTS2*NUM_EVOL_GFS;\n\n"
        self.body += self.device_kernel.launch_block
        self.body += self.device_kernel.c_function_call()
        self.CFunction = cfc.CFunction(
            prefunc=prefunc,
            includes=self.includes,
            desc=self.desc,
            cfunc_type=self.cfunc_type,
            name=self.name,
            params=self.params,
            body=self.body,
        )


# single_RK_substep_input_symbolic() performs necessary replacements to
#   define C code for a single RK substep
#   (e.g., computing k_1 and then updating the outer boundaries)
def single_RK_substep_input_symbolic(
    substep_time_offset_dt: Union[sp.Basic, int, str],
    rhs_str: str,
    rhs_input_expr: sp.Basic,
    rhs_output_expr: sp.Basic,
    RK_lhs_list: Union[sp.Basic, List[sp.Basic]],
    RK_rhs_list: Union[sp.Basic, List[sp.Basic]],
    post_rhs_list: Union[str, List[str]],
    post_rhs_output_list: Union[sp.Basic, List[sp.Basic]],
    rk_step: Union[int, None] = None,
    enable_intrinsics: bool = False,
    gf_aliases: str = "",
    post_post_rhs_string: str = "",
    additional_comments: str = "",
    rational_const_alias: str = "const",
) -> str:
    """
    Generate C code for a given Runge-Kutta substep.

    :param substep_time_offset_dt: Time offset for the RK substep.
    :param rhs_str: Right-hand side string of the C code.
    :param rhs_input_expr: Input expression for the RHS.
    :param rhs_output_expr: Output expression for the RHS.
    :param RK_lhs_list: List of LHS expressions for RK.
    :param RK_rhs_list: List of RHS expressions for RK.
    :param post_rhs_list: List of post-RHS expressions.
    :param post_rhs_output_list: List of outputs for post-RHS expressions.
    :param rk_step: Optional integer representing the current RK step.
    :param enable_intrinsics: Whether hardware intrinsics are enabled.
    :param gf_aliases: Additional aliases for grid functions.
    :param post_post_rhs_string: String to be used after the post-RHS phase.
    :param additional_comments: additional comments to append to auto-generated comment block.
    :param rational_const_alias: Provide additional/alternative alias to const for rational definitions

    :return: A string containing the generated C code.

    :raises ValueError: If substep_time_offset_dt cannot be extracted from the Butcher table.
    """
    # Ensure all input lists are lists
    RK_lhs_list = [RK_lhs_list] if not isinstance(RK_lhs_list, list) else RK_lhs_list
    RK_rhs_list = [RK_rhs_list] if not isinstance(RK_rhs_list, list) else RK_rhs_list
    post_rhs_list = (
        [post_rhs_list] if not isinstance(post_rhs_list, list) else post_rhs_list
    )
    post_rhs_output_list = (
        [post_rhs_output_list]
        if not isinstance(post_rhs_output_list, list)
        else post_rhs_output_list
    )
    comment_block = (
        f"// -={{ START k{rk_step} substep }}=-"
        if not rk_step is None
        else "// ***Euler timestepping only requires one RHS evaluation***"
    )
    comment_block += additional_comments
    body = f"{comment_block}\n"

    if isinstance(substep_time_offset_dt, (int, sp.Rational, sp.Mul)):
        substep_time_offset_str = f"{float(substep_time_offset_dt):.17e}"
    else:
        raise ValueError(
            f"Could not extract substep_time_offset_dt={substep_time_offset_dt} from Butcher table"
        )
    body += "for(int grid=0; grid<commondata->NUMGRIDS; grid++) {\n"
    body += (
        f"commondata->time = time_start + {substep_time_offset_str} * commondata->dt;\n"
    )
    body += gf_aliases

    # Part 1: RHS evaluation
    updated_rhs_str = (
        str(rhs_str)
        .replace("RK_INPUT_GFS", str(rhs_input_expr).replace("gfsL", "gfs"))
        .replace("RK_OUTPUT_GFS", str(rhs_output_expr).replace("gfsL", "gfs"))
    )
    body += updated_rhs_str + "\n"

    # Part 2: RK update
    RK_key = f"RK_STEP{rk_step}"
    base_MoL.MoL_Functions_dict[RK_key] = RKFunction(
        "REAL",
        RK_lhs_list,
        RK_rhs_list,
        rk_step=rk_step,
        enable_intrinsics=enable_intrinsics,
        rational_const_alias=rational_const_alias,
    )

    body += base_MoL.MoL_Functions_dict[RK_key].c_function_call()

    # Part 3: Call post-RHS functions
    for post_rhs, post_rhs_output in zip(post_rhs_list, post_rhs_output_list):
        body += post_rhs.replace(
            "RK_OUTPUT_GFS", str(post_rhs_output).replace("gfsL", "gfs")
        )

    body += "}\n"

    for post_rhs, post_rhs_output in zip(post_rhs_list, post_rhs_output_list):
        body += post_post_rhs_string.replace(
            "RK_OUTPUT_GFS", str(post_rhs_output).replace("gfsL", "gfs")
        )

    return body


########################################################################################################################
# EXAMPLE
# ODE: y' = f(t,y), y(t_0) = y_0
# Starting at time t_n with solution having value y_n and trying to update to y_nplus1 with timestep dt

# Example of scheme for RK4 with k_1, k_2, k_3, k_4 (Using non-diagonal algorithm) Notice this requires storage of
# y_n, y_nplus1, k_1 through k_4

# k_1      = dt*f(t_n, y_n)
# k_2      = dt*f(t_n + 1/2*dt, y_n + 1/2*k_1)
# k_3      = dt*f(t_n + 1/2*dt, y_n + 1/2*k_2)
# k_4      = dt*f(t_n + dt, y_n + k_3)
# y_nplus1 = y_n + 1/3k_1 + 1/6k_2 + 1/6k_3 + 1/3k_4

# Example of scheme RK4 using only k_odd and k_even (Diagonal algroithm) Notice that this only requires storage


# k_odd     = dt*f(t_n, y_n)
# y_nplus1  = 1/3*k_odd
# k_even    = dt*f(t_n + 1/2*dt, y_n + 1/2*k_odd)
# y_nplus1 += 1/6*k_even
# k_odd     = dt*f(t_n + 1/2*dt, y_n + 1/2*k_even)
# y_nplus1 += 1/6*k_odd
# k_even    = dt*f(t_n + dt, y_n + k_odd)
# y_nplus1 += 1/3*k_even
########################################################################################################################
class register_CFunction_MoL_step_forward_in_time(
    base_MoL.base_register_CFunction_MoL_step_forward_in_time
):
    r"""
    Register MoL_step_forward_in_time() C function, which is the core driver for time evolution in BHaH codes.

    :param Butcher_dict: A dictionary containing the Butcher tables for various RK-like methods.
    :param MoL_method: The method of lines (MoL) used for time-stepping.
    :param rhs_string: Right-hand side string of the C code.
    :param post_rhs_string: Input string for post-RHS phase in the C code.
    :param post_post_rhs_string: String to be used after the post-RHS phase.
    :param enable_rfm_precompute: Flag to enable reference metric functionality.
    :param enable_curviBCs: Flag to enable curvilinear boundary conditions.
    :param enable_intrinsics: Whether hardware intrinsics are enabled.

    DOCTEST:
    >>> import nrpy.c_function as cfc, json
    >>> from nrpy.infrastructures.gpu.MoLtimestepping.base_MoL import MoL_Functions_dict
    >>> from nrpy.helpers.generic import validate_strings
    >>> from nrpy.infrastructures.BHaH.MoLtimestepping.RK_Butcher_Table_Dictionary import generate_Butcher_tables
    >>> Butcher_dict = generate_Butcher_tables()
    >>> rhs_string = "rhs_eval(commondata, params, rfmstruct,  auxevol_gfs, RK_INPUT_GFS, RK_OUTPUT_GFS);"
    >>> post_rhs_string=(
    ... "if (strncmp(commondata->outer_bc_type, \"extrapolation\", 50) == 0)\n"
    ... "  apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, RK_OUTPUT_GFS);"
    ... )
    >>> for k, v in Butcher_dict.items():
    ...     Butcher = Butcher_dict[k][0]
    ...     cfc.CFunction_dict.clear()
    ...     MoL_Functions_dict.clear()
    ...     if Butcher[-1][0] != "":
    ...         continue
    ...     _MoLclass = register_CFunction_MoL_step_forward_in_time(
    ...         Butcher_dict,
    ...         k,
    ...         rhs_string=rhs_string,
    ...         post_rhs_string=post_rhs_string
    ...     )
    ...     generated_str = cfc.CFunction_dict["MoL_step_forward_in_time"].full_function
    ...     validation_desc = f"CUDA__MoL_step_forward_in_time__{k}".replace(" ", "_")
    ...     validate_strings(generated_str, validation_desc)
    >>> cfc.CFunction_dict.clear()
    >>> MoL_Functions_dict.clear()
    >>> try:
    ...     register_CFunction_MoL_step_forward_in_time(Butcher_dict, "AHE")
    ... except ValueError as e:
    ...     print(f"ValueError: {e.args[0]}")
    ValueError: Adaptive order Butcher tables are currently not supported in MoL.
    """

    def __init__(
        self,
        Butcher_dict: Dict[str, Tuple[List[List[Union[sp.Basic, int, str]]], int]],
        MoL_method: str,
        rhs_string: str = "",
        post_rhs_string: str = "",
        post_post_rhs_string: str = "",
        enable_rfm_precompute: bool = False,
        enable_curviBCs: bool = False,
        enable_intrinsics: bool = False,
        rational_const_alias: str = "static constexpr",
    ) -> None:

        super().__init__(
            Butcher_dict,
            MoL_method,
            rhs_string=rhs_string,
            post_rhs_string=post_rhs_string,
            post_post_rhs_string=post_post_rhs_string,
            enable_rfm_precompute=enable_rfm_precompute,
            enable_curviBCs=enable_curviBCs,
            enable_intrinsics=enable_intrinsics,
            rational_const_alias=rational_const_alias,
        )
        if self.enable_intrinsics:
            self.includes += [os.path.join("intrinsics", "cuda_intrinsics.h")]
        self.single_RK_substep_input_symbolic = single_RK_substep_input_symbolic
        self.gf_alias_prefix = "MAYBE_UNUSED"
        self.setup_gf_aliases()
        self.generate_RK_steps()

        self.register()


# register_CFunction_MoL_free_memory() registers
#           MoL_free_memory_y_n_gfs() and
#           MoL_free_memory_non_y_n_gfs(), which free memory for
#           the indicated sets of gridfunctions
class register_CFunction_MoL_free_memory(
    base_MoL.base_register_CFunction_MoL_free_memory
):
    """
    Free memory for the specified Method of Lines (MoL) gridfunctions, given an MoL_method.

    :param Butcher_dict: Dictionary containing Butcher tableau for MoL methods.
    :param MoL_method: The Method of Lines method.
    :param which_gfs: The gridfunctions to be freed, either 'y_n_gfs' or 'non_y_n_gfs'.
    """

    def __init__(
        self,
        Butcher_dict: Dict[str, Tuple[List[List[Union[sp.Basic, int, str]]], int]],
        MoL_method: str,
        which_gfs: str,
    ) -> None:

        super().__init__(Butcher_dict, MoL_method, which_gfs)
        for gridfunction in self.gridfunctions_list:
            # Don't free a zero-sized array.
            if gridfunction == "auxevol_gfs":
                self.body += (
                    f"  if(NUM_AUXEVOL_GFS > 0) cudaFree(gridfuncs->{gridfunction});"
                )
            else:
                self.body += f"  cudaFree(gridfuncs->{gridfunction});"

        self.register()


# Register all the CFunctions and NRPy basic defines
class register_CFunctions(base_MoL.base_register_CFunctions):
    r"""
    Register all MoL C functions and NRPy basic defines.

    :param MoL_method: The method to be used for MoL. Default is 'RK4'.
    :param rhs_string: RHS function call as string. Default is "rhs_eval(Nxx, Nxx_plus_2NGHOSTS, dxx, RK_INPUT_GFS, RK_OUTPUT_GFS);"
    :param post_rhs_string: Post-RHS function call as string. Default is "apply_bcs(Nxx, Nxx_plus_2NGHOSTS, RK_OUTPUT_GFS);"
    :param post_post_rhs_string: Post-post-RHS function call as string. Default is an empty string.
    :param enable_rfm_precompute: Enable reference metric support. Default is False.
    :param enable_curviBCs: Enable curvilinear boundary conditions. Default is False.
    :param enable_intrinsics: Whether hardware intrinsics are enabled. Default is False.
    :param register_MoL_step_forward_in_time: Whether to register the MoL step forward function. Default is True.

    Doctests:
    >>> from nrpy.helpers.generic import validate_strings
    >>> cfc.CFunction_dict.clear()
    >>> _ = register_CFunctions()
    >>> generated_str = cfc.CFunction_dict["MoL_step_forward_in_time"].full_function
    >>> validate_strings(generated_str, f"CUDA__MoL_step_forward_in_time")
    >>> sorted(cfc.CFunction_dict.keys())
    ['MoL_free_memory_non_y_n_gfs', 'MoL_free_memory_y_n_gfs', 'MoL_malloc_non_y_n_gfs', 'MoL_malloc_y_n_gfs', 'MoL_step_forward_in_time']
    >>> print(cfc.CFunction_dict["MoL_free_memory_non_y_n_gfs"].full_function)
    #include "BHaH_defines.h"
    #include "BHaH_function_prototypes.h"
    /**
     * Method of Lines (MoL) for "RK4" method: Free memory for "non_y_n_gfs" gridfunctions
     * - y_n_gfs are used to store data for the vector of gridfunctions y_i at t_n, at the start of each MoL timestep
     * - non_y_n_gfs are needed for intermediate (e.g., k_i) storage in chosen MoL method
     *
     */
    void MoL_free_memory_non_y_n_gfs(MoL_gridfunctions_struct *restrict gridfuncs) {
      cudaFree(gridfuncs->y_nplus1_running_total_gfs);
      cudaFree(gridfuncs->k_odd_gfs);
      cudaFree(gridfuncs->k_even_gfs);
      if (NUM_AUXEVOL_GFS > 0)
        cudaFree(gridfuncs->auxevol_gfs);
    }
    <BLANKLINE>
    >>> print(cfc.CFunction_dict["MoL_malloc_non_y_n_gfs"].full_function)
    #include "BHaH_defines.h"
    #include "BHaH_function_prototypes.h"
    /**
     * Method of Lines (MoL) for "RK4" method: Allocate memory for "non_y_n_gfs" gridfunctions
     * - y_n_gfs are used to store data for the vector of gridfunctions y_i at t_n, at the start of each MoL timestep
     * - non_y_n_gfs are needed for intermediate (e.g., k_i) storage in chosen MoL method
     */
    void MoL_malloc_non_y_n_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                MoL_gridfunctions_struct *restrict gridfuncs) {
    #include "set_CodeParameters.h"
      const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
      cudaMalloc(&gridfuncs->y_nplus1_running_total_gfs, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
      cudaCheckErrors(malloc, "Malloc failed");
      cudaMalloc(&gridfuncs->k_odd_gfs, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
      cudaCheckErrors(malloc, "Malloc failed");
      cudaMalloc(&gridfuncs->k_even_gfs, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
      cudaCheckErrors(malloc, "Malloc failed");
      if (NUM_AUXEVOL_GFS > 0)
        cudaMalloc(&gridfuncs->auxevol_gfs, sizeof(REAL) * NUM_AUXEVOL_GFS * Nxx_plus_2NGHOSTS_tot);
      cudaCheckErrors(malloc, "Malloc failed");
    <BLANKLINE>
      gridfuncs->diagnostic_output_gfs = gridfuncs->y_nplus1_running_total_gfs;
      gridfuncs->diagnostic_output_gfs2 = gridfuncs->k_odd_gfs;
    }
    <BLANKLINE>
    """

    def __init__(
        self,
        MoL_method: str = "RK4",
        rhs_string: str = "rhs_eval(Nxx, Nxx_plus_2NGHOSTS, dxx, RK_INPUT_GFS, RK_OUTPUT_GFS);",
        post_rhs_string: str = "apply_bcs(Nxx, Nxx_plus_2NGHOSTS, RK_OUTPUT_GFS);",
        post_post_rhs_string: str = "",
        enable_rfm_precompute: bool = False,
        enable_curviBCs: bool = False,
        enable_intrinsics: bool = False,
        register_MoL_step_forward_in_time: bool = True,
    ) -> None:
        super().__init__(
            MoL_method=MoL_method,
            rhs_string=rhs_string,
            post_rhs_string=post_rhs_string,
            post_post_rhs_string=post_post_rhs_string,
            enable_rfm_precompute=enable_rfm_precompute,
            enable_curviBCs=enable_curviBCs,
            register_MoL_step_forward_in_time=register_MoL_step_forward_in_time,
        )
        for which_gfs in ["y_n_gfs", "non_y_n_gfs"]:
            register_CFunction_MoL_malloc(self.Butcher_dict, MoL_method, which_gfs)
            register_CFunction_MoL_free_memory(self.Butcher_dict, MoL_method, which_gfs)
        if register_MoL_step_forward_in_time:
            register_CFunction_MoL_step_forward_in_time(
                self.Butcher_dict,
                self.MoL_method,
                self.rhs_string,
                post_rhs_string=post_rhs_string,
                post_post_rhs_string=post_post_rhs_string,
                enable_rfm_precompute=self.enable_rfm_precompute,
                enable_curviBCs=self.enable_curviBCs,
                enable_intrinsics=enable_intrinsics,
            )

        griddata_commondata.register_griddata_commondata(
            __name__, "MoL_gridfunctions_struct gridfuncs", "MoL gridfunctions"
        )

        # Step 3.b: Create MoL_timestepping struct:
        self.BHaH_MoL_body = self.BHaH_MoL_body.replace("REAL *restrict ", "REAL * ")

        BHaH_defines_h.register_BHaH_defines(__name__, self.BHaH_MoL_body)


if __name__ == "__main__":
    import doctest
    import sys

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
