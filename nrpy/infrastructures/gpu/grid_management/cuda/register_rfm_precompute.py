"""
CUDA implementation to register CFunctions for precomputed reference metric infrastructure.

Authors: Samuel D. Tootle
        sdtootle **at** gmail **dot** com
        Zachariah B. Etienne
        zachetie **at** gmail **dot** com
"""

from typing import List

import nrpy.helpers.gpu.gpu_kernel as gputils
from nrpy.helpers.expression_utils import get_unique_expression_symbols_as_strings
from nrpy.infrastructures.gpu.grid_management.base_register_rfm_precompute import (
    base_register_CFunctions_rfm_precompute,
)
from nrpy.infrastructures.gpu.grid_management.cuda.rfm_precompute import (
    ReferenceMetricPrecompute,
)


class register_CFunctions_rfm_precompute(base_register_CFunctions_rfm_precompute):
    """
    Cuda implementation to register C functions for reference metric precomputed lookup arrays.

    :param list_of_CoordSystems: List of coordinate systems to register the C functions.
    """

    def __init__(self, list_of_CoordSystems: List[str]) -> None:
        super().__init__(list_of_CoordSystems)

        # Overload with CUDA implementation
        self.rfm_class = ReferenceMetricPrecompute
        self.generate_rfm_core_functions()
        self.generate_rfm_CUDA_kernels()
        self.register()

    def generate_rfm_CUDA_kernels(self) -> None:
        """Generate CUDA kernels for RFM precompute."""
        for CoordSystem in self.list_of_CoordSystems:
            rfm_precompute = ReferenceMetricPrecompute(CoordSystem)

            for func, kernel_dicts in [
                ("defines", rfm_precompute.rfm_struct__define_kernel_dict),
            ]:

                desc = f"rfm_precompute_{func}: reference metric precomputed lookup arrays: {func}"
                cfunc_type = "void"
                name = "rfm_precompute_" + func
                params = "const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct"
                params += ", REAL *restrict xx[3]"

                body = " "
                for i in range(3):
                    body += f"MAYBE_UNUSED const REAL *restrict x{i} = xx[{i}];\n"
                prefunc = ""
                prefunc_defs = ""
                for i, (key_sym, kernel_dict) in enumerate(kernel_dicts.items()):
                    # prefunc_defs += f"REAL *restrict {key_sym} = rfmstruct->{key_sym};\n"
                    # These should all be in paramstruct?
                    unique_symbols = get_unique_expression_symbols_as_strings(
                        kernel_dict["expr"], exclude=[f"xx{j}" for j in range(3)]
                    )
                    kernel_body = ""
                    kernel_body += "// Temporary parameters\n"
                    for sym in unique_symbols:
                        kernel_body += f"const REAL {sym} = d_params[streamid].{sym};\n"
                    kernel_body += kernel_dict["body"]
                    device_kernel = gputils.GPU_Kernel(
                        kernel_body,
                        {
                            f"{key_sym}": "REAL *restrict",
                            f'{kernel_dict["coord"]}': "const REAL *restrict",
                        },
                        f"{name}__{key_sym}_gpu",
                        launch_dict={
                            "blocks_per_grid": [],
                            "threads_per_block": ["32"],
                            "stream": f"(param_streamid + {i}) % nstreams",
                        },
                        comments=f"GPU Kernel to precompute metric quantity {key_sym}.",
                    )
                    prefunc += device_kernel.CFunction.full_function
                    body += "{\n"
                    body += f"REAL *restrict {key_sym} = rfmstruct->{key_sym};\n"
                    body += (
                        "const size_t param_streamid = params->grid_idx % nstreams;\n"
                    )
                    body += device_kernel.launch_block
                    body += device_kernel.c_function_call().replace(
                        "(streamid", "(param_streamid"
                    )
                    body += "}\n"
                body = prefunc_defs + body

                self.function_dict[name] = {
                    "desc": desc,
                    "cfunc_type": cfunc_type,
                    "params": params,
                    "body": body,
                    "CoordSystem": CoordSystem,
                    "prefunc": prefunc,
                    "include_CodeParameters_h": True,
                }
