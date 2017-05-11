#!/usr/local/bin/env python

import os
import glob

import yaml
import numpy as np

from yank import mpi
from yank.analyze import get_analyzer
from yank.yamlbuild import YamlBuilder


# A validation test fails when its Z-score exceeds this threshold.
MAX_Z_SCORE = 6


def run_validation():
    """Run all validation tests."""
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
        kT = analysis.kT

    # Compute free energy.
    DeltaF = 0.0
    dDeltaF = 0.0
    for phase_name, sign in analysis:
        DeltaF -= sign * (analysis[phase_name]['DeltaF'] + analysis[phase_name]['DeltaF_standard_state_correction'])
        dDeltaF += analysis[phase_name]['dDeltaF']**2
    dDeltaF = np.sqrt(dDeltaF)

    return DeltaF, dDeltaF


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
    expected_DeltaF = expected_free_energy[0]
    expected_dDeltaF = expected_free_energy[1]
    obtained_DeltaF = obtained_free_energy[0]

    # Determine if test has passed.
    z_score = (obtained_DeltaF - expected_DeltaF) / expected_dDeltaF
    test_passed = abs(z_score) < MAX_Z_SCORE

    # Print results.
    print('{}: {}\n'
          '\texpected: {} * kT\n'
          '\tobtained: {} * kT\n'
          '\tZ-score {}:'.format(experiment_name, 'OK' if test_passed else 'FAIL',
                                 expected_free_energy,
                                 obtained_free_energy,
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
        run_experiments_names = glob.glob(os.path.join(testset_dir, 'experiments', '*'))

        # Distribute analysis of all experiments across nodes.
        free_energies = mpi.distribute(analyze_directory, run_experiments_names,
                                       send_results_to='all')

        # Print comparisons.
        for experiment_id, experiment_name in run_experiments_names:
            print_analysis(experiment_name, expected_output[experiment_name], free_energies[experiment_id])


if __name__ == '__main__':
    run_validation()
    run_analysis()
