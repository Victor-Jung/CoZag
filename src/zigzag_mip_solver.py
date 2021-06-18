import argparse
import logging
import os
import pathlib
import time

import numpy as np
import run_config

from gurobipy import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # capture everything
logger.disabled = True

cosa_dir = os.environ['COSA_DIR']


def zigzag_mip_solver(f, strides, arch, part_ratios, global_buf_idx, A, B, compute_factor=10, util_factor=-1,
               traffic_factor=1):
    """CoSA mixed integer programming(MIP) formulation."""

    logging.info(f"LAYER {f}")

    num_vars = len(A[0])
    num_mems = len(B[0])

    m = Model("mip")
    cost = []
    constraints = []

    org = ['spatial', 'temporal']

    M = []

    # ignore DRAM cap
    for i in range(num_mems - 1):
        mem_cap = arch.mem_entries[i]
        mem_cap_arr = []
        for j in range(num_vars):
            var_mem_cap = mem_cap * part_ratios[i][j]
            mem_cap_arr.append(var_mem_cap)
        M.append(mem_cap_arr)

    # log friendly M
    M_log = []
    for i, mem in enumerate(M):
        M_v = []
        for bound in mem:
            if bound == 0:
                # turn 0 to 1 for taking the log
                bound = 1
            M_v.append(bound)
        M_log.append(M_v)

    # spatial constraints
    S = arch.S

    # set the levels to be equal to the number of factors + 4 memory levels 
    perm_levels = 0
    for j, f_j in enumerate(f):
        perm_levels += len(f_j)
    gb_start_level = global_buf_idx

    total_levels = num_mems - 1 + perm_levels
    logger.info(f"total {total_levels} levels")

    ### Create Z matrix, giving the memory level of each loop for each data tensor (W/I/O) ###

    Z = np.zeros((perm_levels, num_vars, num_mems), dtype="int")
    M_copy = np.ones((num_mems, num_vars), dtype="int")
    factor_list = [] # factor list sorted growing order
    current_mem_level = 0 

    # List factors in f and count the number of loop for each factors
    for j, f_j in enumerate(f):
        for n, f_jn in enumerate(f_j):
            factor_list.append(f_jn)
    factor_list.sort()

    for i, factor in enumerate(factor_list):
        for v in range(num_vars):
            for mem_level in range(num_mems):
                if mem_level == num_mems - 1: # Last Mem Level, we assume inf capacity
                    Z[i][v][mem_level] = 1
                elif M[mem_level][v] >= factor*M_copy[mem_level][v]:
                    Z[i][v][mem_level] = 1
                    M_copy[mem_level][v] *= factor
                    break
                
    ### Create X matrix, linking each factor to his order and spatial or temporal attribution ###

    x = {}  # x_jn_jn
    for i in range(perm_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    x[(i, j, n, k)] = m.addVar(vtype=GRB.BINARY, name=name)
                # Constraint : Each loop have to be attributed spatially or temporally
                spatial_temp_sum = 0
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    spatial_temp_sum += x[(i, j, n, k)]
                m.addConstr(spatial_temp_sum == 1, "spatial_temp_sum_{}_{}_{}".format(i, j, n))

    # Constraint : Each order should be linked to one and one factor only
    i = 0
    x_row_sums = []
    x_col_sums = []
    for i in range(perm_levels):
        row_sum = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    row_sum += x[(i, j, n, k)]
        m.addConstr(row_sum == 1, "row_sum_{}".format(i))
        x_row_sums.append(row_sum)

    # Constraint : Each factor should be linked to one and one order only (spatially or temporally)
    for j, f_j in enumerate(f):
        for n, f_jn in enumerate(f_j):
            col_sum = 0
            for i in range(perm_levels):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    col_sum += x[(i, j, n, k)]
            # assume perm can be interleaved in diff perm level
            m.addConstr(col_sum == 1, "col_sum_{}_{}".format(j, n))
            x_col_sums.append(col_sum)

            # make sure v is one for all outer loop level, once a correlation exists
    # add another relation var v - f, 3 - 7*n loop-level
    s = {}
    y = {}
    for v in range(num_vars):
        for i in range(perm_levels):
            row_sum = 0
            y[(v, i)] = m.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name="y({},{})".format(v, i))
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    row_sum += x[(i, j, n, 1)] * A[j][v] # Why no B marix here ?
            if i > gb_start_level:
                m.addConstr(y[(v, i)] >= y[(v, i - 1)], "y_v_i_sv_{}_{}".format(v, i))
                # can be ==
                m.addConstr(y[(v, i)] >= row_sum, "y_v_i_row_sum_{}_{}".format(v, i))
            else:
                # can be ==
                m.addConstr(y[(v, i)] == row_sum, "y_v_i_row_sum_{}_{}".format(v, i))
            s[(v, i)] = row_sum

    ## exhausively list all scenarios where p or q is inside current mem
    zz = {}
    prefix = 0
    for var in [2, 3]:
        for mem_level in [3]:
            zz[(var, mem_level)] = m.addVar(lb=0, ub=1, vtype=GRB.INTEGER,
                                            name="zz({},{},{})".format(prefix, var, mem_level))
            x_sums = 0
            for n, prime_factor in enumerate(f[var]):
                for inner_mem_level_i in range(mem_level + 1):
                    for k in range(2):
                        filter_in = x[(inner_mem_level_i, var, n, k)]
                        m.addConstr(zz[(var, mem_level)] >= filter_in,
                                    "zz_x_sum_{}_{}_{}_{}_{}_{}".format(prefix, var, n, mem_level, inner_mem_level_i,
                                                                        k))
                        x_sums += filter_in
            m.addConstr(zz[(var, mem_level)] <= x_sums, "z_x_sum_{}_{}_{}".format(prefix, var, mem_level))

    l = {}
    for v in range(num_vars):
        for i in range(perm_levels):
            row_sum = 0
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    row_sum += np.log2(f[j][n]) * (x[(i, j, n, 1)])
            l[(v, i)] = row_sum

    # Add spatial constraints
    spatial_tile = 0
    for i in range(perm_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                spatial_tile += np.log2(f[j][n]) * x[(i, j, n, 0)]
    m.addConstr(spatial_tile <= np.log2(S[gb_start_level]), "spatial_tile_gb_{}".format(prefix))

    for i in range(gb_start_level):
        spatial_tile = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                spatial_tile += np.log2(f[j][n]) * x[(i, j, n, 0)]
        m.addConstr(spatial_tile <= np.log2(S[i]), f"spatial_tile_{prefix}_{i}")

    for i in range(perm_levels):
        spatial_tile = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                spatial_tile += np.log2(f[j][n]) * x[(i, j, n, 0)]
        m.addConstr(spatial_tile <= np.log2(S[i - perm_levels + 1]), f"spatial_tile_{i - perm_levels + 1}")

    # Add inner gb buffer constraints
    buf_util = {}
    # memory id
    for i in range(gb_start_level):
        # for each memory and each variable there is a constraint
        for v in range(num_vars):
            size = 0
            for i_ in range(num_mems):
                for j, f_j in enumerate(f):
                    for n, f_jn in enumerate(f_j):
                        factor = 1
                        if v == 1 and j == 2: 
                            factor = strides[0]
                        if v == 1 and j == 3: 
                            factor = strides[1]
                        size += np.log2(factor * f[j][n]) * (x[(i_,j,n,0)] + x[i_,j,n,1]) * A[j][v] * B[v][i] # use the i for the cur mem for relationship 
                        # only add once
                        if i == 3 and j in [0,1] and v == 1: 
                            size += (x[(i_,j,n,0)]+x[(i_,j,n,1)]) * (1-zz[(j+2, i)]) * np.log2(f[j][n]) 
                            size += (x[(i_,j,n,0)]+x[(i_,j,n,1)]) * zz[(j+2, i)] * np.log2(2)
            if B[v][i]:
                m.addConstr(size <= np.log2(M_log[i][v]), f"buffer_size_{i}_{v}")
            buf_util[(i, v)] = size
    
    # Add gb buffer constraints 
    for v in range(num_vars):
        size = 0 
        for i in range(perm_levels):
            for i_ in range(num_mems):
                for j, f_j in enumerate(f):
                    for n, f_jn in enumerate(f_j):
                        size +=  np.log2(f[j][n]) * (x[(i,j,n,0)] + x[(i,j,n,1)]) * A[j][v] * B[v][gb_start_level]

        if B[v][gb_start_level] :
            m.addConstr(size <= np.log2(M_log[gb_start_level][v]), f"buffer_size_{gb_start_level}_{v}")
        buf_util[(gb_start_level, v)] = size

    '''
    # Add inner gb buffer constraints
    buf_util = {}
    for v in range(num_vars):
        for i in range(num_mems):
            buf_util[(i, v)] = 0

    for v in range(num_vars):
        for i_ in range(gb_start_level + perm_levels):
            for i in range(num_mems):
                for j, f_j in enumerate(f):
                    for n, f_jn in enumerate(f_j):
                        factor = 1
                        if v == 1 and j == 2:
                            factor = strides[0]
                        if v == 1 and j == 3:
                            factor = strides[1]

                        if i_ > gb_start_level and i_ < gb_start_level + perm_levels:
                            Z_const = Z[v][i][gb_start_level]
                        else:
                            Z_const = Z[v][i][i_]
                        buf_util[(i, v)] += np.log2(factor * f[j][n]) * (x[(i_, j, n, 0)] + x[i_, j, n, 1]) * A[j][v] # use the i for the cur mem for relationship 
                        # only add once
                        if i == 3 and j in [0, 1] and v == 1:
                            buf_util[(i, v)] += (x[(i_, j, n, 0)] + x[(i_, j, n, 1)]) * (1 - zz[(j + 2, i)]) * np.log2(f[j][n])
                            buf_util[(i, v)] += (x[(i_, j, n, 0)] + x[(i_, j, n, 1)]) * zz[(j + 2, i)] * np.log2(2)

    for v in range(num_vars):
        # excluding DRAM
        for i in range(num_mems - 1):
            if M_log[i][v] > 0:
                m.addConstr(buf_util[(i, v)] <= np.log2(M_log[i][v]), f"buffer_size_{i}_{v}")'''

    # get compute cost 
    inner_gb_cycles = 0
    for i in range(gb_start_level):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                inner_gb_cycles += np.log2(f[j][n]) * (x[(i, j, n, 1)])

    gb_cycles = 0
    for i in range(perm_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                gb_cycles += np.log2(f[j][n]) * (x[(i, j, n, 1)])

    dram_cycles = 0
    for i in range(perm_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                dram_cycles += np.log2(f[j][n]) * (x[(i, j, n, 1)])
    total_compute = inner_gb_cycles + gb_cycles + dram_cycles
    gb_compute = inner_gb_cycles + gb_cycles

    # get traffic cost
    spatial_cost = {}
    for v in range(num_vars):
        size = 0
        for i in range(perm_levels):
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    size += np.log2(f[j][n]) * (x[(i, j, n, 0)])
        spatial_cost[v] = size

    data_size = {}
    for v in range(num_vars):
        size = 0
        for i in range(gb_start_level):
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    # TRICK prioritize spatial
                    factors = 0.8 + 0.04 * i
                    size += factors * np.log2(f[j][n]) * (x[(i, j, n, 0)] + x[i, j, n, 1]) * A[j][v]
        data_size[v] = size

    gb_traffic = {}
    for v in range(num_vars):
        size = 0
        for i in range(perm_levels):
            size += l[(v, i)] * y[(v, i)]
        gb_traffic[v] = size

        # use the last level gb y for DRAM 
    dram_traffic = {}
    for v in range(num_vars):
        corr = y[(v, perm_levels)]
        i = gb_start_level + perm_levels  # DRAM 
        size = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                size += np.log2(f[j][n]) * (x[(i, j, n, 1)])  # * corr 
        dram_traffic[v] = size

    total_util = 0
    for i in range(gb_start_level):
        # for each memory and each variable there is a constraint
        for v in range(num_vars):
            # make weight util more important since it directly comes from dram
            factor = 1.01 if i == 2 else 1
            total_util += buf_util[(i, v)] * factor

    total_traffic = 0
    for v in range(num_vars):
        #  TRICKS
        if v == 0:
            # encode dram latency for weights
            factor = 1.01
        else:
            factor = 1

        total_traffic += 0.99 * data_size[v] + 0.99 * spatial_cost[v] + gb_traffic[v] + dram_traffic[v] * factor

    # ========================== user-defined objective function ========================== #
    cosa_obj = total_util * util_factor + total_compute * compute_factor + total_traffic * traffic_factor

    max_it = m.addVar(vtype=GRB.CONTINUOUS, name="max_it")
    its = []
    its.append(m.addVar(vtype=GRB.CONTINUOUS, name="a"))
    m.addConstr(its[-1] == total_traffic, "total_traffic")
    its.append(m.addVar(vtype=GRB.CONTINUOUS, name="b"))
    m.addConstr(its[-1] == total_compute, "total_compute")
    m.addConstr(max_it == max_(its), name="max_it_constr")

    total_util_var = m.addVar(vtype=GRB.CONTINUOUS, name="total_util_var")
    total_comp_var = m.addVar(vtype=GRB.CONTINUOUS, name="total_comp_var")
    total_traf_var = m.addVar(vtype=GRB.CONTINUOUS, name="total_traf_var")

    # cycle count = total max 3 * all log factors variables 
    m.addConstr(total_util_var == total_util, "total_util_constraint")
    m.addConstr(total_comp_var == total_compute, "total_comp_constraint")
    m.addConstr(total_traf_var == total_traffic, "total_traf_constraint")

    m.ModelSense = GRB.MINIMIZE
    m.setObjective(cosa_obj, GRB.MINIMIZE)

    # optimize for the objective function
    milp_time = 0
    begin_time = time.time()
    m.optimize()
    end_time = time.time()
    milp_runtime = end_time - begin_time

    # output all constraints and variables
    m.write("debug.lp")

    result_dict = {}
    for variable in m.getVars():
        logger.debug("Variable %s: value %s" % (variable.varName, variable.x))
        assert (variable.varName not in result_dict)
        result_dict[variable.varName] = variable.x
    logger.debug('Obj: %g' % m.objVal)

    all_x = np.zeros((total_levels, perm_levels, 2))
    for i in range(total_levels):
        level_idx = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    val = result_dict[name]
                    all_x[i, level_idx, k] = val
                level_idx += 1
    np.set_printoptions(precision=0, suppress=True)

    var_outer_perm_config = [-1] * perm_levels
    outer_perm_config = [-1] * perm_levels
    x_arr = np.zeros((perm_levels, perm_levels, 2))
    for i in range(gb_start_level, gb_start_level + perm_levels):
        level_idx = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    val = result_dict[name]
                    x_arr[i - gb_start_level, level_idx, k] = val
                name = "X({},{},{},{})".format(i, j, n, 1)
                val = result_dict[name]
                if val == 1:
                    var_outer_perm_config[i - gb_start_level] = j
                level_idx += 1
    logger.info(f'var_outer_perm_config: {var_outer_perm_config}')

    y_arr = np.zeros((num_vars, perm_levels))
    for v in range(num_vars):
        for i in range(gb_start_level, gb_start_level + perm_levels):
            row_sum = 0
            val = result_dict["y({},{})".format(v, i)]
            y_arr[v, i - gb_start_level] = val
    # logger.info(x_arr.reshape(perm_levels, -1))

    # Merge the permutation, taking the first appearance of a prob to be the
    merge_outer_perm_config = []
    for i, var in enumerate(var_outer_perm_config):
        if var != -1 and var not in merge_outer_perm_config:
            merge_outer_perm_config.append(var)

    for i in range(len(f)):
        if i not in merge_outer_perm_config:
            merge_outer_perm_config.append(i)

    logger.info("var idx as the value {}".format(var_outer_perm_config))
    logger.info("merged var idx as the value {}".format(merge_outer_perm_config))

    outer_perm_config = [1] * len(f)
    for i, var in enumerate(merge_outer_perm_config):
        outer_perm_config[var] = i

    logger.info("ordering idx as the value {}".format(outer_perm_config))

    # init factor_config 
    # DRAM is the last level
    factor_config = []
    spatial_config = []
    dram_level = -1
    for j, f_j in enumerate(f):
        sub_factor_config = []
        sub_spatial_config = []
        for n, f_jn in enumerate(f_j):
            sub_factor_config.append(dram_level)
            sub_spatial_config.append(1)
        factor_config.append(sub_factor_config)
        spatial_config.append(sub_spatial_config)

    for i in range(gb_start_level):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                if f[j][n] == 1:
                    factor_config[j][n] = num_mems - 1
                    continue
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    val = result_dict[name]
                    if val >= 0.9:
                        factor_config[j][n] = i
                        if k == 0:
                            spatial_config[j][n] = 0

    for i in range(gb_start_level + perm_levels, total_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    val = result_dict[name]
                    if k == 0:
                        if val >= 0.9:
                            factor_config[j][n] = spatial_to_factor_map[i]
                            assert False
                    else:
                        if val >= 0.9:
                            factor_config[j][n] = i - perm_levels + 1

    # set to -1 for not specified 
    for j, f_j in enumerate(f):
        for n, f_jn in enumerate(f_j):
            for i in range(gb_start_level, gb_start_level + perm_levels):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    val = result_dict[name]
                    if val >= 0.9:
                        factor_config[j][n] = gb_start_level
                        if k == 0:
                            spatial_config[j][n] = 0

    logger.info(f"gb factors: {f}")
    logger.info(f"factor_config: {factor_config}")
    logger.info(f"gb spatial_config: {spatial_config}")

    level_idx = 0
    for j, f_j in enumerate(f):
        for n, f_jn in enumerate(f_j):
            if factor_config[j][n] == -1:
                logger.info(all_x[:][idx])
                level_idx += 1
                # TRICK rough fix to make it dram        
                assert False

    # factor_config = factor_config_opt_double_buf(factor_config, f)
    return (factor_config, spatial_config, outer_perm_config, milp_runtime)