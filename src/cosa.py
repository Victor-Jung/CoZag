#!/usr/bin/env python3 
import argparse
import logging
import os
import pathlib
import time

import numpy as np
import run_config
from cosa_constants import _A, _B
from cosa_input_objs import Prob, Arch, Mapspace
from timeloop_mip_solver import timeloop_mip_solver
from zigzag_mip_solver import zigzag_mip_solver
from gurobipy import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # capture everything
logger.disabled = True

cosa_dir = os.environ['COSA_DIR']


def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Output Folder',
                        default='output_dir',
                        )
    parser.add_argument('-a',
                        '--arch_path',
                        type=str,
                        help='Hardware Architecture Path',
                        default=f'{cosa_dir}/configs/arch/simba.yaml',
                        )
    parser.add_argument('-m',
                        '--mapspace_path',
                        type=str,
                        help='Mapspace Path',
                        default=f'{cosa_dir}/configs/mapspace/mapspace.yaml',
                        )
    parser.add_argument('-p',
                        '--prob_path',
                        type=str,
                        help='Problem Dimension Path',
                        default=f'{cosa_dir}/configs/workloads/resnet50_graph/_outputs_input.2.yaml',
                        )
    return parser


def cosa(prob, arch, A, B, part_ratios, global_buf_idx, Z=None):
    """Run CoSA to generate a mapping with tiling, temporal/spatial, and permutation determined. 
        We currently assume there is a global buffer and only perform the permutation optimization
        at the global buffer level. Will add perm to all level in future version. 

    Args:
        prob: An object defines the layer dimension.
        arch: An object defines the hardware architecture dimension. 
        A: A 2d binary constant matrix that encodes the layer dimension to data tensor relationship.
            1 means related, 0 means unrelated
            Note that the R,S to the input tensor relation is specially handled in the formulation,
            and are specified to 0. 
        B: A 2d binary constant matrix that encodes the data tensor to memory level mapping. 
            It can be derived from the mapspace bypass pattern in Timeloop. 
            Note it is intended to be used for even mapping among different data tensors to different memory levels.
        part_ratios: A 2d array to represent the partition ratios of different data tensors 
            in different memory buffers. 
        global_buf_idx: An index point to the global buffer. 
        Z: Similar to B, but intended for uneven mapping among different data tensors to different memory levels.
            It is a 3d binary constant matrix that encodes the data tensor to memory level mapping.

    Returns: 
        factor_config: A 2d array specifying the allocation decision for each prime factor.
        spatial_config: A 2d array specifying the temporal/spatial decisions for each prime factor.
        perm_config: A 2d array specifyng the ordering of R,S,P,Q,C,K,N factors at each level.  
        run_time: Time-to-solution of CoSA.
    """
    # prime factors 
    prime_factors = prob.prob_factors
    strides = [prob.prob['Wstride'], prob.prob['Hstride']]

    framework = 'zigzag'

    if framework == 'zigzag':

        factor_config, spatial_config, outer_perm_config, run_time = zigzag_mip_solver(prime_factors, strides, arch, part_ratios,
                                                                                global_buf_idx=4, A=A, B=B,
                                                                                compute_factor=10, util_factor=-0.1,
                                                                                traffic_factor=1)

    elif framework == 'timeloop':

        if Z is None:
            Z = []
            for var in _B:
                Z_var = []
                for i, val in enumerate(var):
                    rank_arr = [0] * len(var)
                    if val == 1:
                        for j in range(i + 1):
                            rank_arr[j] = 1
                    Z_var.append(rank_arr)
                Z.append(Z_var)

        factor_config, spatial_config, outer_perm_config, run_time = timeloop_mip_solver(prime_factors, strides, arch, part_ratios,
                                                                                global_buf_idx=4, A=_A, Z=Z,
                                                                                compute_factor=10, util_factor=-0.1,
                                                                                traffic_factor=1)
    else:
        raise Exception('Unknown framework name')

    return factor_config, spatial_config, outer_perm_config, run_time


def run_timeloop(args):
    output_path = args.output_dir
    arch_path = pathlib.Path(args.arch_path).resolve()
    mapspace_path = pathlib.Path(args.mapspace_path).resolve()
    prob_path = pathlib.Path(args.prob_path).resolve()

    # init
    status_dict = {}
    prob = Prob(prob_path)
    arch = Arch(arch_path)

    # An object defines the user-defined bypass pattern. 
    mapspace = Mapspace(mapspace_path)
    mapspace.init(prob, arch)

    # even mapping
    B = _B
    Z = None

    # uneven mapping config
    # Z = _Z
    # B = None

    # partition ratios for W, IA, OA
    part_ratios = [
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0.25, 0.75],
        [0.33, 0.33, 0.33],
    ]
    factor_config, spatial_config, outer_perm_config, run_time = cosa(prob, arch, _A, B, part_ratios, global_buf_idx=4,
                                                                      Z=Z)

    update_factor_config = factor_config
    spatial_to_factor_map = {}
    idx = arch.mem_levels
    for i, val in enumerate(arch.S):
        if val > 1:
            spatial_to_factor_map[i] = idx
            idx += 1
    logger.info(f'spatial_to_factor_map: {spatial_to_factor_map}')

    for j, f_j in enumerate(prob.prob_factors):
        for n, f_jn in enumerate(f_j):
            # if is mapped to spatial, look up the combined index
            if spatial_config[j][n] == 0:
                idx = factor_config[j][n]
                update_factor_config[j][n] = spatial_to_factor_map[idx]

    perm_config = mapspace.get_default_perm()
    perm_config[4] = outer_perm_config
    try:
        spatial_configs = []
        status_dict = {}
        results = run_config.run_config(mapspace, None, perm_config, update_factor_config, status_dict,
                                        run_gen_map=True, run_gen_tc=False, run_sim_test=False, output_path=output_path,
                                        spatial_configs=spatial_configs, valid_check=False, outer_loopcount_limit=100)
        logger.info(f'status_dict: {status_dict}')
    except:
        raise


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()
    run_timeloop(args)
