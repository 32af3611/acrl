#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:05:14 2022
"""

import errno
import os.path
import shutil
import subprocess

from src.rl.applications.cfd.lib import util_data


def cfd_blackbox(root_dir, case_desc, blc_loc_ss, blc_coef_ss, blc_loc_ps, blc_coef_ps, cores, verbose=True):
    """
    creates, runs and postprocesses a case of given parameters

    parameters:
        root_dir:  data root directory
        case_id:   unique string to identify case (will also be used as case directory name)
        case_desc: string to describe the case
        blc_loc:  locations   : n+1 coordinates for x interval of blowing/suction on suction side ("_SS") or pressure side ("_PS") of airfoil
        blc_coef: coefficients: n floating point numbers to set blowing/suction on suction side ("_SS") or pressure side ("_PS") of airfoil
                        --> positive Values: suction
                        --> negative Values: blowing
    """
    output_args = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if not verbose else dict()

    here = os.path.dirname(__file__)
    copy_anything(f'{here}/RL003', root_dir)
    # prepare funcySetBoundary dict:
    set_funky_set_boundary_dict(root_dir, blc_loc_ss, blc_coef_ss, blc_loc_ps, blc_coef_ps)

    subprocess.Popen(["taskset", "-c", cores, "bash", "./Allrun"], cwd=root_dir, shell=False, **output_args).wait()
    drag_coefficient, iterations = post_process_case(root_dir)
    return drag_coefficient, iterations


def copy_anything(src, dst):
    """
    Copy dictionary from src to dest including all subdicts and files
    """
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else:
            raise


def set_funky_set_boundary_dict(case_dir, blc_loc_ss, blc_coef_ss, blc_loc_ps, blc_coef_ps):
    """
    set the boundary layer control configuration
    
    parameters:
        blc_loc:  locations   : n+1 coordinates for x interval of blowing/suction
        blc_coef: coefficients: n floating point numbers to set blowing/suction 
        
        Meaning (Example):
            i       : 1     2     ...    n
            blc_loc : 0.2   0.25  ...    0.8    0.85
            blc_coef: 0.001 0.002 ...    0.0005
   
            --> positive Values: suction
            --> negative Values: blowing
    """
    # Open FunkySetBoundaryDict
    dictname = case_dir + '/system/funkySetBoundaryDict'
    templatename = case_dir + '/system/funkySetBoundaryDict_template'

    with open(templatename, 'rb') as fin:
        orig_string = fin.read()
        append_string = orig_string.decode()

    # Set manipulation fields
    locs = [blc_loc_ss, blc_loc_ps]
    coefs = [blc_coef_ss, blc_coef_ps]
    patch_names = ['suction_side', 'pressure_side']
    for i_Patch in range(len(locs)):
        # get arrays of the current patch
        blc_loc = locs[i_Patch]
        blc_coef = coefs[i_Patch]
        patch_name = patch_names[i_Patch]

        # check if there is a BLC present on the current patch
        if len(blc_coef) > 0:
            # There is a BLC configuration requested for this patch
            # header:
            append_string = append_string + 'blc_velocity_' + patch_name + '\n'
            append_string = append_string + '{\n'
            append_string = append_string + '\tfield U;\n'
            append_string = append_string + '\texpressions\n'
            append_string = append_string + '\t(\n'

            # generate funky entry
            append_string = append_string + '\t\t{\n'
            append_string = append_string + '\t\t\ttarget value;\n'
            append_string = append_string + '\t\t\tpatchName ' + patch_name + ';\n'
            append_string = append_string + '\t\t\texpression "'

            for i_Coef in range(len(blc_coef)):
                x_loc_start = blc_loc[i_Coef]
                x_loc_end = blc_loc[i_Coef + 1]
                v_blow = blc_coef[i_Coef]

                append_string = append_string + get_funky_expression(x_loc_start, x_loc_end, v_blow)

            # Set non_manipulated surface to no manipulation (Velocity vector= (0,0,0))
            append_string = append_string + ' vector(0,0,0)";\n'
            append_string = append_string + '\t\t}\n'

            # Set Foot
            append_string = append_string + '\t);\n'
            append_string = append_string + '}\n\n'

    # Write to file
    with open(dictname, 'wb') as fout:
        data = append_string.encode()
        fout.write(data)


def post_process_case(case_path):
    """
    postProcesses a case, returns drag coefficient
    """
    # generate local quantity report
    try:
        iteration, x_ss, y_ss, x_ps, y_ps = util_data.read_panel_coordinates(case_path)
        alpha_deg, cl, cd, cd_p, cd_f, cm, x_rel_ss, cp_ss, cf_ss, x_rel_ps, cp_ps, cf_ps = util_data.post_readCase(case_path)
        return cd, iteration
    except FileNotFoundError as fnfe:
        print('Case ' + case_path + ' Surface length cannot be generated as there is at least one file missing: ' + str(fnfe))
        return None


def get_funky_expression(x_loc_start, x_loc_end, v_blow):
    return 'pos().x>' + str(x_loc_start) + ' && pos().x<' + str(x_loc_end) + ' ? normal()*' + str(v_blow) + ' : '
