#!/usr/local/bin/env python

import os
import sys
import glob
import json
import logging

import yaml
import numpy as np

from simtk import unit

from yank import mpi
from yank.analyze import get_analyzer
from yank.yamlbuild import YamlBuilder


# A validation test fails when its Z-score exceeds this threshold.
MAX_Z_SCORE = 6

# Set verbosity.
logging.basicConfig(level=logging.DEBUG)


def run_validation():
    """Run all validation tests.

    This is probably best done by running the different validation set
    singularly since the optimal number of GPUs depends on the protocol.

    """
    for yank_script_filepath in glob.glob(os.path.join('..', '*', '*.yaml')):
        print('Running {}...'.format(os.path.basename(yank_script_filepath)))
        yaml_builder = YamlBuilder(yank_script_filepath)
        yaml_builder.run_experiments()


def analyze_directory(experiment_dir):
    """Return free energy and error of a single experiment.

    Parameters
    ----------
    experiment_dir : str
        The path to the directory storing the nc files.

    Return
    ------
    DeltaF : simtk.unit.Quantity
        Difference in free energy between the end states in kcal/mol.
    dDeltaF: simtk.unit.Quantity
        Statistical error of the free energy estimate.

    """
    print('Analyzing {}'.format(experiment_dir))
    sys.stdout.flush()
    analysis_script_filepath = os.path.join(experiment_dir, 'analysis.yaml')

    # Load sign of alchemical phases.
    with open(analysis_script_filepath, 'r') as f:
        analysis_script = yaml.load(f)

     # Generate analysis object.
    analysis = {}
    for phase_name, sign in analysis_script:
        phase_path = os.path.join(experiment_dir, phase_name + '.nc')
        analyzer = get_analyzer(phase_path)
        analysis[phase_name] = analyzer.analyze_phase()
        kT = analyzer.kT

    # Compute free energy.
    DeltaF = 0.0
    dDeltaF = 0.0
    for phase_name, sign in analysis_script:
        DeltaF -= sign * (analysis[phase_name]['DeltaF'] + analysis[phase_name]['DeltaF_standard_state_correction'])
        dDeltaF += analysis[phase_name]['dDeltaF']**2
    dDeltaF = np.sqrt(dDeltaF)

    # Convert from kT units to kcal/mol
    unit_conversion = kT / unit.kilocalories_per_mole
    return DeltaF * unit_conversion, dDeltaF * unit_conversion


@mpi.on_single_node(0)
def print_analysis(experiment_name, expected_free_energy, obtained_free_energy):
    """Print the results of the analysis.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment to print.
    expected_free_energy : tuple of float
        The expected pair (DeltaF, dDeltaF) in kT.
    obtained_free_energy : tuple of float
        The pair (DeltaF, dDeltaF) in kT from the calculation.

    """
    expected_DeltaF, expected_dDeltaF = expected_free_energy
    obtained_DeltaF, obtained_dDeltaF = obtained_free_energy

    # Determine if test has passed.
    z_score = (obtained_DeltaF - expected_DeltaF) / expected_dDeltaF
    test_passed = abs(z_score) < MAX_Z_SCORE

    # Print results.
    print('{}: {}\n'
          '\texpected: {:.3f} +- {:.3f} kcal/mol\n'
          '\tobtained: {:.3f} +- {:.3f} kcal/mol\n'
          '\tZ-score: {:.3f}'.format(experiment_name, 'OK' if test_passed else 'FAIL',
                                 expected_DeltaF, expected_dDeltaF,
                                 obtained_DeltaF, obtained_dDeltaF,
                                 z_score))


def run_analysis():
    """Run analysis on all validation tests."""
    for expected_output_filepath in glob.glob(os.path.join('..', '*', 'expected_output.yaml')):

        # Load expected results.
        # expected_output is a dictionary experiment_name: [DeltaF, dDeltaF]
        with open(expected_output_filepath, 'r') as f:
            expected_output = yaml.load(f)

        # Find all experiments that we have run so far.
        testset_dir = os.path.dirname(expected_output_filepath)
        run_experiments_paths = glob.glob(os.path.join(testset_dir, 'experiments', '*'))

        # Distribute analysis of all experiments across nodes.
        free_energies = mpi.distribute(analyze_directory, run_experiments_paths,
                                       send_results_to='all', group_nodes=1)

        # Store and print results.
        analysis_summary = dict()
        for experiment_id, experiment_path in enumerate(run_experiments_paths):
            experiment_name = os.path.basename(experiment_path)
            analysis_summary[experiment_name] = [expected_output[experiment_name], free_energies[experiment_id]]
            print_analysis(experiment_name, expected_output[experiment_name], free_energies[experiment_id])
        with open('analysis_summary.json', 'w') as f:
            json.dump(analysis_summary, f)


if __name__ == '__main__':
    # run_validation()
    run_analysis()
