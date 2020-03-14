#!/usr/local/bin/env python

import os
import copy
import pickle
import logging
import collections

import numpy as np
import pymbar
import openmmtools as mmtools

from simtk import openmm, unit
from yank import restraints, experiment, mpi


logger = logging.getLogger(__name__)


def restrain_atoms(thermodynamic_state, sampler_state, mdtraj_topology,
                   atoms_dsl, sigma=3.0*unit.angstroms):
    K = thermodynamic_state.kT / sigma  # Spring constant.
    system = thermodynamic_state.system  # This is a copy.

    # Translate the system to the origin to avoid
    # MonteCarloBarostat rejections (see openmm#1854).
    protein_atoms = mdtraj_topology.select('protein')
    distance_unit = sampler_state.positions.unit
    centroid = np.mean(sampler_state.positions[protein_atoms,:] / distance_unit, 0) * distance_unit
    sampler_state.positions -= centroid

    # Create a CustomExternalForce to restrain all atoms.
    restraint_force = openmm.CustomExternalForce('(K/2)*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
    restrained_atoms = mdtraj_topology.select(atoms_dsl).tolist()
    # Adding the spring constant as a global parameter allows us to turn it off if desired
    restraint_force.addGlobalParameter('K', K)
    restraint_force.addPerParticleParameter('x0')
    restraint_force.addPerParticleParameter('y0')
    restraint_force.addPerParticleParameter('z0')
    for index in restrained_atoms:
        parameters = sampler_state.positions[index,:].value_in_unit_system(unit.md_unit_system)
        restraint_force.addParticle(index, parameters)

    # Update thermodynamic state.
    system.addForce(restraint_force)
    thermodynamic_state.system = system


def save_optimization_info(optimization_info, name):
    import pickle
    pickle_file_path1 = 'optimization_info_{}_1.pkl'.format(name)
    pickle_file_path2 = 'optimization_info_{}_2.pkl'.format(name)
    if os.path.exists(pickle_file_path1):
        optimization_info_path = pickle_file_path2
    else:
        optimization_info_path = pickle_file_path1
    with open(optimization_info_path, 'wb') as f:
        pickle.dump(optimization_info, f)


def find_optimal_protocol_exaustive(thermodynamic_state, sampler_state, mcmc_move,
                                    state_parameters, std_energy_threshold=1.0, n_samples_per_state=100,
                                    name=''):
    # Make sure that the state parameters to optimize have a clear order.
    assert(isinstance(state_parameters, list) or isinstance(state_parameters, tuple))

    # Initialize protocol with starting values.
    optimal_protocol = {par[0]: [par[1][0]] for par in state_parameters}
    optimization_info = []

    # We change only one parameter at a time.
    for state_parameter, values in state_parameters:
        current_value_index = 0  # The index of value being reweighted.

        # Gather data until we get to the last value.
        while optimal_protocol[state_parameter][-1] != values[-1]:
            # Gather energy datapoints.
            sampler_states = []
            energies = np.zeros(n_samples_per_state)
            for i in range(n_samples_per_state):
                mcmc_move.apply(thermodynamic_state, sampler_state)
                sampler_states.append(copy.deepcopy(sampler_state))
                energies[i] = thermodynamic_state.reduced_potential(sampler_state)

            # Find first state that doesn't overlap with current one.
            # TODO handle units here?
            std_energy = 0.0
            while std_energy < std_energy_threshold:
                # If the last state has good overlap, we're done.
                if current_value_index == len(values) - 1:
                    current_value_index += 1
                    break

                # Modify thermodynamic state.
                current_value_index += 1
                current_value = values[current_value_index]
                setattr(thermodynamic_state, state_parameter, current_value)

                # Get context in new thermodynamic state.
                context, integrator = mmtools.cache.global_context_cache.get_context(thermodynamic_state)

                # Compute energies for all positions.
                reweighted_energies = np.zeros(n_samples_per_state)
                for i, sampler_state in enumerate(sampler_states):
                    sampler_state.apply_to_context(context, ignore_velocities=True)
                    reweighted_energies[i] = thermodynamic_state.reduced_potential(context)

                # Compute standard deviation of the difference.
                denergies = reweighted_energies - energies
                std_energy = np.std(denergies)
                df, ddf = pymbar.EXP(denergies)
                optimization_info.append({
                    'state_parameter': state_parameter,
                    'simulated_value': optimal_protocol[state_parameter][-1],
                    'current_value': current_value,
                    'std_du': std_energy,
                    'exp_df': df,
                    'exp_ddf': ddf,
                    'simulated_energies': energies,
                    'reweighted_energies': reweighted_energies
                })
                printed_info = copy.deepcopy(optimization_info[-1])
                printed_info.pop('simulated_energies')
                printed_info.pop('reweighted_energies')
                print(printed_info)
                logger.info(printed_info)

            # Check that found state is not the old one.
            current_value_index -= 1  # The current state has poor overlap.
            new_value = values[current_value_index]
            if optimal_protocol[state_parameter][-1] == new_value:
                raise RuntimeError('Could not find a non overlapping state for {} from {} to {}.\n'
                                   'Std: {}\tDF: {}\t DDF: {}\nEnergies: {}\nReweighted energies: {}'.format(
                    state_parameter, new_value, values[current_value_index+1],
                    std_energy, df, ddf, energies, reweighted_energies))

            # Update the optimal protocol with the new value of this parameter.
            # The other parameters remain fixed.
            for par_name in optimal_protocol:
                if par_name == state_parameter:
                    optimal_protocol[par_name].append(new_value)
                else:
                    optimal_protocol[par_name].append(optimal_protocol[par_name][-1])

    save_optimization_info(optimization_info, name)
    return optimal_protocol


def find_optimal_protocol_newton(thermodynamic_state, sampler_state, mcmc_move, state_parameters,
                                 std_energy_threshold=0.5, threshold_tolerance=0.05,
                                 n_samples_per_state=100, name=''):

    def compute_std_energy(parameter_value):
        """Return energy standard deviation for the given parameter."""
        # Get context in new thermodynamic state.
        setattr(thermodynamic_state, state_parameter, parameter_value)
        context, integrator = mmtools.cache.global_context_cache.get_context(thermodynamic_state)

        reweighted_energies = np.zeros(n_samples_per_state)
        for i, sampler_state in enumerate(sampler_states):
            sampler_state.apply_to_context(context, ignore_velocities=True)
            reweighted_energies[i] = thermodynamic_state.reduced_potential(context)

        # Compute standard deviation of the difference.
        denergies = reweighted_energies - simulated_energies
        std_energy = np.std(denergies)
        df, ddf = pymbar.EXP(denergies)
        optimization_info.append({
            'state_parameter': state_parameter,
            'simulated_value': optimal_protocol[state_parameter][-1],
            'tested_value': parameter_value,
            'std_du': std_energy,
            'exp_df': df,
            'exp_ddf': ddf,
            'simulated_energies': simulated_energies,
            'reweighted_energies': reweighted_energies
        })
        printed_info = copy.deepcopy(optimization_info[-1])
        printed_info.pop('simulated_energies')
        printed_info.pop('reweighted_energies')
        print(printed_info)
        return std_energy

    # Make sure that the state parameters to optimize have a clear order.
    assert(isinstance(state_parameters, list) or isinstance(state_parameters, tuple))
    optimization_info = []  # TODO find a better way to store optimization info.

    # Make sure that thermodynamic_state is in correct state
    # and initialize protocol with starting value.
    optimal_protocol = {}
    for parameter, values in state_parameters:
        setattr(thermodynamic_state, parameter, values[0])
        optimal_protocol[parameter] = [values[0]]

    # We change only one parameter at a time.
    for state_parameter, values in state_parameters:
        # Is this a search from 0 to 1 or from 1 to 0?
        search_direction = np.sign(values[1] - values[0])
        # If the parameter doesn't change, continue to the next one.
        if search_direction == 0:
            continue

        # Gather data until we get to the last value.
        while optimal_protocol[state_parameter][-1] != values[-1]:
            # Simulate current thermodynamic state to obtain energies.
            sampler_states = []
            simulated_energies = np.zeros(n_samples_per_state)
            for i in range(n_samples_per_state):
                mcmc_move.apply(thermodynamic_state, sampler_state)
                sampler_states.append(copy.deepcopy(sampler_state))
                simulated_energies[i] = thermodynamic_state.reduced_potential(sampler_state)

            # Find first state that doesn't overlap with simulated one
            # with std(du) within std_energy_threshold +- threshold_tolerance.
            # We stop anyway if we reach the last value of the protocol.
            std_energy = 0.0
            current_parameter_value = optimal_protocol[state_parameter][-1]
            while (abs(std_energy - std_energy_threshold) > threshold_tolerance and
                   not (current_parameter_value == values[1] and std_energy < std_energy_threshold)):
                # Determine next parameter value to compute.
                if np.isclose(std_energy, 0.0):
                    # This is the first iteration or the two state overlap significantly
                    # (e.g. small molecule in vacuum). Just advance by a +- 0.05 step.
                    old_parameter_value = current_parameter_value
                    current_parameter_value += (values[1] - values[0]) / 20.0
                else:
                    # Assume std_energy(parameter_value) is linear to determine next value to try.
                    derivative_std_energy = ((std_energy - old_std_energy) /
                                             (current_parameter_value - old_parameter_value))
                    old_parameter_value = current_parameter_value
                    current_parameter_value += (std_energy_threshold - std_energy) / derivative_std_energy

                # Update old std energy value.
                old_std_energy = std_energy

                # Keep current_parameter_value inside bound interval.
                if search_direction * current_parameter_value > values[1]:
                    current_parameter_value = values[1]
                assert search_direction * (optimal_protocol[state_parameter][-1] - current_parameter_value) < 0

                # Compute the standard deviation of du.
                std_energy = compute_std_energy(current_parameter_value)

            # Update the optimal protocol with the new value of this parameter.
            # The other parameters remain fixed.
            for par_name in optimal_protocol:
                if par_name == state_parameter:
                    optimal_protocol[par_name].append(current_parameter_value)
                else:
                    optimal_protocol[par_name].append(optimal_protocol[par_name][-1])

    save_optimization_info(optimization_info, name)
    return optimal_protocol


def prepare_state_parameters_newton(phase):
    """Prepare state_parmeters argument for find_optimal_protocol_newton()."""
    state_parameters = []
    is_vacuum = len(phase.topography.receptor_atoms) == 0 and len(phase.topography.solvent_atoms) == 0

    # We may need to slowly turn on a Boresch restraint.
    if isinstance(phase.restraint, restraints.Boresch):
        state_parameters.append(('lambda_restraints', [0.0, 1.0]))

    # We support only lambda sterics and electrostatics for now.
    if is_vacuum and not phase.alchemical_regions.annihilate_electrostatics:
        state_parameters.append(('lambda_electrostatics', [1.0, 1.0]))
    else:
        state_parameters.append(('lambda_electrostatics', [1.0, 0.0]))
    if is_vacuum and not phase.alchemical_regions.annihilate_sterics:
        state_parameters.append(('lambda_sterics', [1.0, 1.0]))
    else:
        state_parameters.append(('lambda_sterics', [1.0, 0.0]))

    return state_parameters


def prepare_state_parameters_exaustive(phase):
    """Prepare state_parmeters argument for find_optimal_protocol_exaustive()."""
    restraint_lambda_values = np.linspace(0.0, 1.0, num=101)  # 0.01 spaced array of values from 0 to 1.
    electrostatics_lambda_values = np.array(list(reversed(np.linspace(0.0, 1.0, num=101))))
    sterics_lambda_values = np.array(list(reversed(np.linspace(0.0, 1.0, num=1001))))
    assert restraint_lambda_values[0] == 0.0 and restraint_lambda_values[-1] == 1.0  # check for truncation errors.
    assert electrostatics_lambda_values[0] == 1.0 and electrostatics_lambda_values[-1] == 0.0  # check for truncation errors.
    assert sterics_lambda_values[0] == 1.0 and sterics_lambda_values[-1] == 0.0  # check for truncation errors.

    state_parameters = []
    is_vacuum = len(phase.topography.receptor_atoms) == 0 and len(phase.topography.solvent_atoms) == 0

    # We may need to slowly turn on a Boresch restraint.
    if isinstance(phase.restraint, restraints.Boresch):
        state_parameters.append(('lambda_restraints', restraint_lambda_values))

    # We support only lambda sterics and electrostatics for now.
    if is_vacuum and not phase.alchemical_regions.annihilate_electrostatics:
        state_parameters.append(('lambda_electrostatics', [1.0]))
    else:
        state_parameters.append(('lambda_electrostatics', electrostatics_lambda_values))
    if is_vacuum and not phase.alchemical_regions.annihilate_sterics:
        state_parameters.append(('lambda_sterics', [1.0]))
    else:
        state_parameters.append(('lambda_sterics', sterics_lambda_values))

    return state_parameters


def optimize_yank_path(name, yaml_script_filepath, n_equilibration_iterations=1000,
                       n_samples_per_state=100, constrain_receptor=True):
    expbuilder = experiment.ExperimentBuilder(yaml_script_filepath)

    # Be sure yank output directory is unique.
    temp_output_dir = 'temp_output_' + name
    expbuilder.output_dir = temp_output_dir

    optimal_protocols = {}

    for exp in expbuilder.build_experiments():
        for phase in exp.phases:
            phase_id = phase.storage

            # Determine algorithm to use.
            state_parameters = prepare_state_parameters_exaustive(phase)
            find_optimal_protocol = find_optimal_protocol_exaustive
            # state_parameters = prepare_state_parameters_newton(phase)
            # find_optimal_protocol = find_optimal_protocol_newton

            # We only need to create a single state.
            phase.protocol = {par[0]: [par[1][0]] for par in state_parameters}

            # Remove unsampled state that we don't need for the optimization.
            phase.options['anisotropic_dispersion_correction'] = False

            # Set number of equilibration iterations.
            phase.options['number_of_equilibration_iterations'] = n_equilibration_iterations

            # Create the thermodynamic state exactly as AlchemicalPhase would make it.
            alchemical_phase = phase.initialize_alchemical_phase()

            # Get sampler and thermodynamic state and delete alchemical phase.
            thermodynamic_state = alchemical_phase._sampler._thermodynamic_states[0]
            sampler_state = alchemical_phase._sampler._sampler_states[0]
            mcmc_move = alchemical_phase._sampler.mcmc_moves[0]
            del alchemical_phase

            # Restrain the protein heavy atoms to avoid drastic
            # conformational changes (possibly after equilibration).
            if len(phase.topography.receptor_atoms) != 0 and constrain_receptor:
                restrain_atoms(thermodynamic_state, sampler_state, phase.topography.topology,
                               atoms_dsl='name CA', sigma=3.0*unit.angstroms)

            # Find protocol.
            protocol = find_optimal_protocol(thermodynamic_state, sampler_state, mcmc_move,
                                             state_parameters, std_energy_threshold=0.5,
                                             n_samples_per_state=n_samples_per_state, name=name)
            optimal_protocols[phase_id] = protocol

    # Remove temp output directory.
    # TODO remove comment when stop debugging.
    # shutil.rmtree(temp_output_dir)

    return optimal_protocols


def parallelizable_optimize_yank_path(args):
    name, n_equil_iter, constrain_receptor, n_samples_per_state, yaml_script_filepath = args  # Unpack.
    return optimize_yank_path(name, yaml_script_filepath, n_equil_iter, n_samples_per_state, constrain_receptor)


def read_stddev(optimization_info_path):
    # Read optimization information.
    with open(optimization_info_path, 'rb') as f:
        optimization_info = pickle.load(f)
    pass

def read_protocol(optimization_info_path, yank_format=False, return_stdev=False):
    # Read optimization information.
    with open(optimization_info_path, 'rb') as f:
        optimization_info = pickle.load(f)

    # Extract actual simulated lambda values
    std_du_values = collections.OrderedDict()
    lambda_values = collections.OrderedDict()
    for frame in optimization_info:
        std_du = frame['std_du']
        tested_value = frame['current_value']  # 'tested_value' with Newton's method.
        lambda_value = frame['simulated_value']
        state_parameter = frame['state_parameter']
        try:
            if lambda_values[state_parameter][-1] != lambda_value:
                lambda_values[state_parameter].append(lambda_value)
                std_du_values[state_parameter].append([])
        except KeyError:
            lambda_values[state_parameter] = [lambda_value]
            std_du_values[state_parameter] = [[(tested_value, std_du)]]
        else:
            std_du_values[state_parameter][-1].append((tested_value, std_du))

    # Append a at the end of each 0 each.
    for parameter in lambda_values:
        last_value = 1 - lambda_values[parameter][0]
        lambda_values[parameter].append(last_value)

    # Reformat protocol to be YANK-ready.
    if yank_format:
        total_length = sum([len(values) for values in lambda_values.values()])
        current_length = 0
        for parameter, values in lambda_values.items():
            lambda_values[parameter] = [1.0] * (current_length - 1) + values + [0.0] * (total_length - current_length - len(values) - 1)
            current_length += len(values)

    if return_stdev:
        return lambda_values, std_du_values
    return lambda_values


def print_protocol(protocol):
    # Print protocol.
    for parameter, values in protocol.items():
        pretty_print_values = ', '.join(['{:.3f}'.format(x) for x in values])
        print('{}: [{}]'.format(parameter, pretty_print_values))


def plot_protocols(optimization_info_paths):
    import plotutils
    from matplotlib import pyplot as plt

    # Configure.
    figsize = (14, 8)
    plt.rcParams['font.size'] = 12
    phase_environments = ['T4 Lysozyme L99A', 'TIP3P water']
    colormap_name = 'viridis'
    colormap_values = [0.0, 0.2, 0.9, 1.0]

    # Create a 2 by 2 figure.
    fig, subplot_axes = plt.subplots(ncols=2, nrows=2, figsize=figsize)

    # Set color cycler.
    for ax in subplot_axes.flatten():
        plotutils.set_colormap_cycle(colormap_values=colormap_values, colormap_name=colormap_name, axes=ax)

    legend_labels = []

    for experiment_id, (molecule_name, phase_paths) in enumerate(optimization_info_paths.items()):
        for phase_id, phase_path in enumerate(phase_paths):
            # Read alchemical path and std(du)
            protocol, std_du = read_protocol(phase_path, return_stdev=True)

            # Shortcuts for color.
            # color = colors[2*experiment_id + phase_id]

            # Save label for the legend to be added later.
            legend_labels.append('{} in {}'.format(molecule_name, phase_environments[phase_id]))

            # Plot protocol/std_du for electrostatics/sterics.
            for state_parameter_id, (state_parameter, lambda_values) in enumerate(protocol.items()):
                # We print only the std(du) of from first lambda value.
                tested_values, std_du_plotted = zip(*std_du[state_parameter][0])
                std_du_plotted = [0.0] + list(std_du_plotted)  # Add self-std
                delta_lambdas = [0.0] + [lambda_values[0] - v for v in tested_values]
                # Plot.
                subplot_axes[state_parameter_id][0].plot(lambda_values, marker='o', markersize=3, linewidth=2)
                subplot_axes[state_parameter_id][1].plot(delta_lambdas, std_du_plotted, marker='o', markersize=3, linewidth=2)

    # Add legends, labels and titles.
    for axes_row, title_name in zip(subplot_axes, ['electrostatics', 'sterics']):
        axes_row[0].set_xlabel('thermodynamic state')
        axes_row[0].set_ylabel('alchemical $\lambda$')
        axes_row[0].set_title('alchemical path ({})'.format(title_name))
        axes_row[1].set_xlabel('$\Delta \lambda$')
        axes_row[1].set_ylabel('$\sigma(\Delta u) / k_BT$')
        axes_row[1].set_title('Distribution distance ({})'.format(title_name))
        for ax in axes_row:
            ax.legend(legend_labels)
            plotutils.prettify(ax)

    plt.tight_layout()
    # plt.show()
    plt.savefig('path_optimization_analysis.pdf')


if __name__ == '__main__':
    # Build args to parallelize.
    args = []

    # for constrain_receptor in [True, False]:
    #     for n_equil_iter in [1000, 500, 250]:
    #         for n_samples_per_state in [200, 100]:
    #             name = 'phenol_{}samples_{}equil'.format(n_samples_per_state, n_equil_iter)
    #             if constrain_receptor:
    #                 name += '_constrained'
    #             # 2 replicas.
    #             for replica_id in range(2):
    #                 args.append((name + '_replica{}'.format(replica_id+1), n_equil_iter,
    #                              constrain_receptor, n_samples_per_state, 'yank_optimization.yaml'))

    constrain_receptor = False
    n_equil_iter = 5
    n_samples_per_state = 5
    for name in ['phenol_explicit', 'benzene_explicit']:#, 'phenol_implicit', 'benzene_implicit']:
        for replica_id in [2]:# range(3):
            args.append((name + '_replica{}'.format(replica_id+1), n_equil_iter,
                         constrain_receptor, n_samples_per_state,
                         'yank_optimization_{}.yaml'.format(name)))

    # # Check if this is an array job.
    # try:
    #     job_id = int(sys.argv[1])
    # except IndexError:
    #     pass
    # else:
    #     n_jobs = int(sys.argv[2])
    #     args = [x for i, x in enumerate(args) if i % n_jobs == job_id - 1]
    #
    # # Run parallel protocol optimization.
    # mpi.distribute(parallelizable_optimize_yank_path, distributed_args=args, group_nodes=1)

    optimization_info_dir = 'optimization_info_exaustive'

    # Plot info
    optimization_info_paths = collections.OrderedDict()
    for optimization_info_filename, _, _, _, _ in args:
        base_optimization_info_path = os.path.join(optimization_info_dir,
                                                   'optimization_info_' + optimization_info_filename)
        molecule_name = optimization_info_filename.split('_')[0]
        optimization_info_paths[molecule_name] = [base_optimization_info_path + suffix
                                                  for suffix in ['_1.pkl', '_2.pkl']]
    plot_protocols(optimization_info_paths)

    # # Print info
    # for optimization_info_filename, _, _, _, _ in args:
    #     optimization_info_path = os.path.join(optimization_info_dir,
    #                                           'optimization_info_' + optimization_info_filename)
    #     protocol = read_protocol(optimization_info_path + '_1.pkl')
    #     # print('{:50s} (#electrostatics: {}, #sterics: {}, #total: {})'.format(
    #     #     optimization_info_filename, len(protocol['lambda_electrostatics']), len(protocol['lambda_sterics']),
    #     #     len(protocol['lambda_electrostatics']) + len(protocol['lambda_sterics'])))
    #     print_protocol(protocol)
    #     protocol = read_protocol(optimization_info_path + '_2.pkl')
    #     print_protocol(protocol)
    #     print()
