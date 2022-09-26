#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 12:49:41 2022

@author: hi205
"""
import math
import os

import numpy

import src.rl.applications.cfd.lib.constant as const


def getLastSavedIteration(path):
    """
    returns the iteration number of the last increment that got saved
    """
    try:
        iter_dirs = os.listdir(path)
    except FileNotFoundError:
        print('FileNotFound: Case ' + path)
        return 0
    iterations = []
    iterationStrings = []
    for d in iter_dirs:
        try:
            num = float(d)
            iterations.append(num)
            iterationStrings.append(d)
        except ValueError:
            pass
    iterations, iterationStrings = zip(*sorted(zip(iterations, iterationStrings)))
    try:
        # print('Last saved iteration in ' + str(path) + ' --> ' + iterationStrings[-1])
        return iterationStrings[-1]
    except IndexError:  # no iteration at all
        return None


def post_readCase(casePath):
    """
    reads the postprocessing files
        - forceCoeff (cl, cd, cm)
        - pressure on suction and pressure-side of airfoil (cp)
        - wallshearstress on suction and pressure side of airfoil (cf)
        
    """
    #  hard coded!!!
    alpha_deg = 5

    # get ca, cl data
    lid = str(getLastSavedIteration(casePath + '/' + const.FORCE_COEFFS_FILEPATH[0])) + '/'
    filename = casePath + '/' + const.FORCE_COEFFS_FILEPATH[0] + lid + const.FORCE_COEFFS_FILEPATH[1]

    cm = None
    cl = None
    cd = None
    cd_p = None
    cd_f = None

    try:
        with open(filename) as fin:
            data = fin.readlines()

            # read in force coeffs
            line = data[-1]
            cm = float(line.split("\t")[1])
            cd = float(line.split("\t")[2])
            cl = float(line.split("\t")[3])
            #print('cl, cd, cm, alpha_deg = ' + str(cl) + ' , ' + str(cd) + ' , ' + str(cm) + ' , ' + str(alpha_deg))
    except FileNotFoundError as fnfe:
        print('Case ' + casePath + ': No cl-, cd-, cm-data available. Requested File missing: ' + fnfe.filename)

        #Postprocessing: cp and cf data:
    x_rel_ps = []
    x_rel_ps1 = []
    x_rel_ss = []
    x_rel_ss1 = []
    y_rel_ps = []
    y_rel_ps1 = []
    y_rel_ss = []
    y_rel_ss1 = []

    cp_ps = []
    cp_ss = []
    cf_ps = []
    cf_ss = []

    data_missing = False
    try:
        # get postprocessing/surfaces/*** path of latest timestep
        post_dir = casePath + '/' + const.POSTPROCESSING_SURFACE_DIRPATH
        lid = getLastSavedIteration(post_dir)
        # print('postprocessing; last timestep in subdir ' + post_dir + ': ' + str(lid))
        post_dir = post_dir + str(lid)
        # get cp data of pressure side

        filename = post_dir + '/' + const.PRESSURE_COEFFS_FILENAME[0]
        try:
            with open(filename) as fin:
                data = fin.readlines()

                for line in data:
                    if not line.startswith('#'):
                        values = line.split(' ')
                        x_rel = float(values[0])
                        y_rel = float(values[1])
                        if not x_rel in x_rel_ps:
                            x_rel_ps.append(x_rel)
                            y_rel_ps.append(y_rel)
                            p = float(values[-1])
                            cp = (p - const.P_INF) / (0.5 * const.U_INF ** 2)  # p and P_INF are normalized by rho_inf, thus dynamic pressure is too
                            cp_ps.append(cp)

                #Sort by x_rel:
                cp_ps = [x for _, x in sorted(zip(x_rel_ps, cp_ps))]
                x_rel_ps = sorted(x_rel_ps)
        except FileNotFoundError as fnfe:
            print('Case ' + casePath + ': No cp data available. Requested File missing: ' + fnfe.filename)
            data_missing = True

        # get cp data of suction side
        filename = post_dir + '/' + const.PRESSURE_COEFFS_FILENAME[1]
        try:
            with open(filename) as fin:
                data = fin.readlines()

                for line in data:
                    if not line.startswith('#'):
                        values = line.split(' ')
                        x_rel = float(values[0])
                        y_rel = float(values[1])
                        if not x_rel in x_rel_ss:
                            x_rel_ss.append(x_rel)
                            y_rel_ss.append(y_rel)
                            p = float(values[-1])
                            cp = (p - const.P_INF) / (0.5 * const.U_INF ** 2)  # p and P_INF are normalized by rho_inf, thus dynamic pressure is too
                            cp_ss.append(cp)

                #Sort by x_rel:
                cp_ss = [x for _, x in sorted(zip(x_rel_ss, cp_ss))]
                x_rel_ss = sorted(x_rel_ss)
        except FileNotFoundError as fnfe:
            print('Case ' + casePath + ': No cp data available. Requested File missing: ' + fnfe.filename)
            data_missing = True

        # get cf data of pressure side
        filename = post_dir + '/' + const.FRICTION_COEFFS_FILENAME[0]
        # cd_f accumulation:

        try:
            with open(filename) as fin:
                data = fin.readlines()

                for line in data:
                    if not line.startswith('#'):
                        values = line.split(' ')
                        x_rel = float(values[0])
                        y_rel = float(values[1])
                        if not x_rel in x_rel_ps1:
                            x_rel_ps1.append(x_rel)
                            y_rel_ps1.append(y_rel)
                            tau_x = float(values[3])
                            tau_y = float(values[4])
                            tau = (tau_x ** 2 + tau_y ** 2) ** .5
                            cf = -tau / (0.5 * const.U_INF ** 2) * numpy.sign(tau_x)  # p and P_INF are normalized by rho_inf, thus dynamic pressure is too
                            cf_ps.append(cf)

                #Sort by x_rel:
                cf_ps = [x for _, x in sorted(zip(x_rel_ps1, cf_ps))]
        except FileNotFoundError as fnfe:
            print('Case ' + casePath + ': No cf data available. Requested File missing: ' + fnfe.filename)
            data_missing = True

        # get cf data of suction side
        filename = post_dir + '/' + const.FRICTION_COEFFS_FILENAME[1]
        try:
            with open(filename) as fin:
                data = fin.readlines()

                for line in data:
                    if not line.startswith('#'):
                        values = line.split(' ')
                        x_rel = float(values[0])
                        y_rel = float(values[1])
                        if not x_rel in x_rel_ss1:
                            x_rel_ss1.append(x_rel)
                            y_rel_ss1.append(y_rel)
                            tau_x = float(values[3])
                            tau_y = float(values[4])
                            tau = (tau_x ** 2 + tau_y ** 2) ** .5
                            cf = -tau / (0.5 * const.U_INF ** 2) * numpy.sign(tau_x)  # p and P_INF are normalized by rho_inf, thus dynamic pressure is too
                            cf_ss.append(cf)

                #Sort by x_rel:
                cf_ss = [x for _, x in sorted(zip(x_rel_ss1, cf_ss))]

        except FileNotFoundError as fnfe:
            print('Case ' + casePath + ': No cf data available. Requested File missing: ' + fnfe.filename)
            data_missing = True

    except FileNotFoundError as f:
        print('Case ' + casePath + ': No postprocessing data available. Missing directory: ' + f.filename)
        data_missing = True

    if not data_missing:
        ppcl, ppcl_p, ppcl_f, ppcd, ppcd_p, ppcd_f = calc_coeffs(alpha_deg, x_rel_ss, y_rel_ss, x_rel_ps, y_rel_ps, cp_ss, cp_ps, cf_ss, cf_ps)

        # calculate a cd_p by cd of open foam and cd_f of postprocessing which are both quite correct
        # This is actually a common approach since cp values are usually not suitable to calculate pressure drag because of numerical noise (M. Drela: XFOIL USER PRIMERdT)
        cd_f = ppcd_f
        cd_p = cd - ppcd_f
    else:
        print('Case ' + casePath + ': Postprocessing data missing')
        cd_f = None
        cd_p = None

    #    print('OPENFOAM values / PP values:')
    #    print('cl   = ' + str('{:.5f}'.format(cl)) + '  / ' + str('{:.5f}'.format(ppcl)))
    #    print('cl_p =          / ' + str('{:.5f}'.format(ppcl_p)))
    #    print('cl_f =          / ' + str('{:.5f}'.format(ppcl_f)))
    #    print('cd   = ' + str('{:.5f}'.format(cd)) + '  / ' + str('{:.5f}'.format(ppcd)))
    #    print('cd_p = ' + str('{:.5f}'.format(cd_p))+'  / ' + str('{:.5f}'.format(ppcd_p)))
    #    print('cd_f =          / ' + str('{:.5f}'.format(ppcd_f)))

    return alpha_deg, cl, cd, cd_p, cd_f, cm, x_rel_ss, cp_ss, cf_ss, x_rel_ps, cp_ps, cf_ps


def calc_coeffs(alpha_deg, x_rel_ss, y_rel_ss, x_rel_ps, y_rel_ps, cp_ss, cp_ps, cf_ss, cf_ps):
    """
    This method calculates the friction drag and the pressure drag using cd and cf values.
    1. cd_f values are stable and correct. cd_p values not. this might be due to poor LE refinement were a lot of cd_p distribution gets determined
    2. cl_p values are stable and correct. cl_f values not. the explanation of 1. fits this quite as well which makes it probable that it is a correct assessment
    """
    # define angle of attack vector:
    u_v = numpy.array([numpy.cos(alpha_deg / 180 * math.pi), numpy.sin(alpha_deg / 180 * math.pi), 0])
    #u_v = numpy.array([1,0,0])
    # define total pressure and friction sums:
    fp = numpy.array([0, 0, 0])
    ff = numpy.array([0, 0, 0])

    # build a loop from TE SS to TE PS
    x_rel_lp = x_rel_ss[::-1] + x_rel_ps[1:]
    y_rel_lp = y_rel_ss[::-1] + y_rel_ps[1:]
    cp_lp = cp_ss[::-1] + cp_ps[1:]
    cf_lp = cf_ss[::-1] + cf_ps[1:]

    # iterate trough loop:
    for i in range(0, len(x_rel_lp) - 1):
        # define tangent:
        et = numpy.array([x_rel_lp[i + 1] - x_rel_lp[i], y_rel_lp[i + 1] - y_rel_lp[i], 0])
        # define ez
        ez = numpy.array([0, 0, 1])
        # define normal:
        en = numpy.cross(et, ez)

        # calculate mitfield value (equivalent to trapez rule integration)
        cp = -(cp_lp[i + 1] + cp_lp[i]) / 2
        cf = (cf_lp[i + 1] + cf_lp[i]) / 2

        #add up forces:
        fp = fp + cp * en
        ff = ff + cf * et * numpy.sign(numpy.dot(et, [1, 0, 0]))

    # calculate friction and pressure drag
    cd_p = numpy.dot(fp, u_v)
    cd_f = numpy.dot(ff, u_v)

    #calculate drag
    cd = cd_p + cd_f

    #calculate lift
    u_l = numpy.cross(ez, u_v)
    cl_p = numpy.dot(u_l, fp)
    cl_f = numpy.dot(u_l, ff)
    cl = cl_p + cl_f

    return cl, cl_p, cl_f, cd, cd_p, cd_f


def read_panel_coordinates(casePath):
    """
    reads the panel coordinates and returns them separated by suction side and pressure side.
    uses the wallshearstress results of surfaces funtion
    raises a FileNotFoundError if the file cannot be found
    """
    # get Last calculated case:
    last_iteration = int(getLastSavedIteration(casePath))
    lid = f"{last_iteration}/"

    # get coordinates of paneling in pism layer in order to determin the wall normals:
    # retrieve coordinates from wall-shear-stress result files
    post_dir = casePath + '/' + const.POSTPROCESSING_SURFACE_DIRPATH + lid

    x_ss = []
    y_ss = []
    x_ps = []
    y_ps = []

    try:
        # read SS coordinates
        full_filepath = post_dir + const.FRICTION_COEFFS_FILENAME[1]
        with open(full_filepath) as fin:
            data = fin.readlines()

            for line in data:
                if not line.startswith('#'):
                    values = line.split(' ')
                    x_rel = float(values[0])
                    y_rel = float(values[1])
                    if not x_rel in x_ss:
                        x_ss.append(x_rel)
                        y_ss.append(y_rel)

            #Sort by x_rel:
            y_ss = [x for _, x in sorted(zip(x_ss, y_ss))]
            x_ss = sorted(x_ss)

        # read PS coordinates
        full_filepath = post_dir + const.FRICTION_COEFFS_FILENAME[0]
        with open(full_filepath) as fin:
            data = fin.readlines()

            for line in data:
                if not line.startswith('#'):
                    values = line.split(' ')
                    x_rel = float(values[0])
                    y_rel = float(values[1])
                    if not x_rel in x_ps:
                        x_ps.append(x_rel)
                        y_ps.append(y_rel)

            #Sort by x_rel:
            y_ps = [x for _, x in sorted(zip(x_ps, y_ps))]
            x_ps = sorted(x_ps)
    except FileNotFoundError as fnfe:
        print('Case ' + casePath + ': No cf data available. Requested File missing: ' + fnfe.filename)
        raise

    return last_iteration, x_ss, y_ss, x_ps, y_ps
