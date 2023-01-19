# -*- coding: utf-8 -*-
"""
Script to generate the waveform recipe file parameters1.txt for a 

Created 30-December-2022

@author: Oliver Nakano-Baker
"""

import os
import sys
import argparse

from ehd_models.square_driver import choose_squares_ver1, choose_squares_ver2
# from gridgraph.patterns import write_ehd_params_file


def ehd_wave_to_string(volts, frequency, velocity, top_string=False):
    """
    Return an SIJ control string for a 25% square wave with the given frequency
    and voltage. Columns are:
    waveform[5==25% square]  Amplitude[V]  Bias[V]  frequency[1/s]
    On_speed[cm/s]  On_accel[100]  circle_speed  circle_accel  z_speed  z_accel
    bump_hold[s]  spiral_pitch  spit1_time  spit1_volts  spit2_time
    spit2_volts  up_distance  down_distance  dispanse_hold_time
    """
    STANDARD_END = "\t100.000\t0.000\t0.000\t0.000\t0.000\t0.000\t0.000\t0.000\t0\t0.000\t0\t0.000\t0.000\t0.000\n"
    if top_string:
        operation = "1\t0.0\t0\t0\t1.000"
    else:
        operation = f"5\t{volts/2:.1f}\t{volts/2:.1f}\t{frequency:.3f}\t{velocity:.3f}"
    return operation + STANDARD_END
    
    
def write_ehd_params_file(waves, directory):
    """widths, volts, ms"""
    operations = [ehd_wave_to_string(0, 0, 0, top_string=True)]
    for microns, volts, seconds in waves:
        OVERLAP = 0.5
        DUTY = 4                           # 1/X duty cycle
        # TODO: do we need to enforce int-valued frequency here?
        frequency = 1 / (DUTY * seconds)  # Hz
        velocity = 1e-3 * microns * OVERLAP * frequency  # [mm/s]

        operations.append(ehd_wave_to_string(volts, frequency, velocity))

    params_file = os.path.join(directory, "Parameters1.txt")
    with open(params_file, 'w') as f:
        for line in operations:
            f.write(line)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vt", type=int,
                        help='''V_threshold used in the print run. Nozzle
                        should be "large", "std", or "sfine")''')
    parser.add_argument("--wt", type=float, default=1.0,
                        help='''w_threshold used in the print run. Nozzle
                        should be "large", "std", or "sfine")''')
    parser.add_argument("--noz", type=str, default="std",
                        help='''nozzle type used in the print run. Nozzle
                        should be "large", "std", or "sfine")''')
    parser.add_argument("--tar", type=str,
                        help='''Tuple of the target line widths, in
                        micrometers, and waveform order, to be added to the
                        recipe file.''')
    parser.add_argument("--ver", type=int, default=-1,
                        help='''Version of model to invoke. Defualt to the most
                        recently-added model version.''')
    parser.add_argument("--classifiers", type=str, default=".",
                        help='''Path to classifier models.''')
    parser.add_argument("--regressors", type=str, default=".",
                        help='''Path to regressor models.''')
    args = parser.parse_args()

    VERSION = [1, 2]
    if args.ver in VERSION:    # I hate this
        ver = args.ver
    else:
        ver = VERSION[args.ver]
    print(f"Using model version {ver}")

    widths = eval(args.tar)

    if ver == 1:                # Gen 1 solution
        waves = choose_squares_ver1(widths, v_thresh=args.vt, w_thresh=args.wt)
        waves = [(widths[i], int(waves[i][0]), waves[i][1]) for i in
             range(len(widths))]
    if ver == 2:
        waves = choose_squares_ver2(widths, v_thresh=args.vt, w_thresh=args.wt,
                                    vis_output='vis_squares.png',
                                    classifiers_dir=args.classifiers,
                                    regressors_dir=args.regressors)
    else:
        raise NotImplementedError(f"generate_waves workflow {f} not implemented.")

    print(f"waves: {waves}")
    write_ehd_params_file(waves, ".")