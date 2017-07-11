#!/usr/local/bin/env python

import os
import copy
import logging

import pymbar
import numpy as np
import openmmtools as mmtools

from simtk import openmm, unit
from yank import restraints, yamlbuild


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


def find_optimal_protocol(thermodynamic_state, sampler_state, mcmc_move,
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

    import pickle
    pickle_file_path1 = 'optimization_info_{}_1.pkl'.format(name)
    pickle_file_path2 = 'optimization_info_{}_2.pkl'.format(name)
    if os.path.exists(pickle_file_path1):
        optimization_info_path = pickle_file_path2
    else:
        optimization_info_path = pickle_file_path1
    with open(optimization_info_path, 'wb') as f:
        pickle.dump(optimization_info, f)
    return optimal_protocol



def main():
    name = 'phenol_restrained_1000eq'
    yamlbuilder = yamlbuild.YamlBuilder('yank_optimization.yaml')
    lambda_values = np.linspace(0.0, 1.0, num=101)  # 0.01 spaced array of values from 0 to 1.
    reversed_lambda_values = np.array(list(reversed(lambda_values)))
    assert lambda_values[0] == 0.0 and lambda_values[-1] == 1.0  # check for truncation errors.

    optimal_protocols = {}

    for experiment in yamlbuilder.build_experiments():
        for phase in experiment.phases:
            phase_id = phase.storage

            # We support only lambda sterics and electrostatics for now.
            state_parameters = [('lambda_electrostatics', reversed_lambda_values),
                                ('lambda_sterics', reversed_lambda_values)]

            # We may also need to slowly turn on a Boresch restraint.
            if isinstance(phase.restraint, restraints.Boresch):
                state_parameters.append(('lambda_restraints', lambda_values))

            # We only need to create a single state.
            phase.protocol = {par[0]: [par[1][0]] for par in state_parameters}

            # Remove unsampled state that we don't need for the optimization.
            phase.options['anisotropic_dispersion_correction'] = False

            # Create the thermodynamic state exactly as AlchemicalPhase would make it.
            alchemical_phase = phase.initialize_alchemical_phase()

            # Get sampler and thermodynamic state and delete alchemical phase.
            thermodynamic_state = alchemical_phase._sampler._thermodynamic_states[0]
            sampler_state = alchemical_phase._sampler._sampler_states[0]
            mcmc_move = alchemical_phase._sampler.mcmc_moves[0]
            del alchemical_phase

            # Restrain the protein heavy atoms to avoid drastic
            # conformational changes (possibly after equilibration).
            restrain_atoms(thermodynamic_state, sampler_state, phase.topography.topology,
                           atoms_dsl='name CA', sigma=3.0*unit.angstroms)

            # Find protocol.
            protocol = find_optimal_protocol(thermodynamic_state, sampler_state,
                                             mcmc_move, state_parameters,
                                             std_energy_threshold=0.5,
                                             n_samples_per_state=100,
                                             name=name)
            optimal_protocols[phase_id] = protocol

    # Print results.
    for phase_id, protocol in optimal_protocols.items():
        logger.info(phase_id, name)
        for parameter_name, values in protocol.items():
            # Limit to two digits to avoid flaots precision limitations.
            pretty_values = ', '.join(['{:.2f}'.format(i) for i in values])
            logger.info('  {:21s}: [{}]'.format(parameter_name, pretty_values))


if __name__ == '__main__':
    main()

    # import pickle
    # with open('optimization_info1.pkl', 'rb') as f:
    #     optimization_info = pickle.load(f)
    #
    # # optimization_info.append({
    # #     'state_parameter': state_parameter,
    # #     'simulated_value': optimal_protocol[state_parameter][-1],
    # #     'current_value': current_value,
    # #     'std_du': std_energy,
    # #     'exp_df': df,
    # #     'exp_ddf': ddf,
    # #     'simulated_energies': energies,
    # #     'reweighted_energies': reweighted_energies
    # # })
    #
    # for info in optimization_info:
    #     # if info['state_parameter'] == 'lambda_electrostatics':
    #     print('{:.2f}->{:.2f}: std(du) {:.3f}, df {:.3f}, ddf {:.3f}'.format(info['simulated_value'], info['current_value'],
    #                                                                          info['std_du'], info['exp_df'], info['exp_ddf']))
