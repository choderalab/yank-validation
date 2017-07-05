#!/usr/local/bin/env python

import copy

import pymbar
import numpy as np
import openmmtools as mmtools

from yank import restraints
from yank.yamlbuild import YamlBuilder


def find_optimal_protocol(thermodynamic_state, sampler_state, mcmc_move,
                          state_parameters, std_energy_threshold=1.0, n_samples_per_state=100):
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
            positions = []
            energies = np.zeros(n_samples_per_state)
            for i in range(n_samples_per_state):
                mcmc_move.apply(thermodynamic_state, sampler_state)
                positions.append(copy.deepcopy(sampler_state.positions))
                energies[i] = thermodynamic_state.reduced_potential(sampler_state)

            # Find first state that doesn't overlap with current one.
            # TODO handle units here?
            # std_energy = 0.0
            # while std_energy < std_energy_threshold:
            df = 0.0
            ddf = 0.0
            while ddf < std_energy_threshold:
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
                for i, pos in enumerate(positions):
                    context.setPositions(pos)
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
                    'bar_df': df,
                    'bar_ddf': ddf,
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
    with open('optimization_info.pkl', 'wb') as f:
        pickle.dump(optimization_info, f)
    return optimal_protocol



def main():
    yamlbuilder = YamlBuilder('yank_optimization.yaml')
    lambda_values = np.arange(0.0, 1.01, 0.01)  # 0.01 spaced array of values from 0 to 1.
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

            # Find protocol.
            protocol = find_optimal_protocol(thermodynamic_state, sampler_state,
                                             mcmc_move, state_parameters,
                                             n_samples_per_state=1000)
            optimal_protocols[phase_id] = protocol

            # Skip vacuum phase while we debug solvent phase.
            break


    # Print results.
    for phase_id, protocol in optimal_protocols.items():
        print(phase_id)
        for parameter_name, values in protocol.items():
            print('\t{}: {}'.format(parameter_name, values))


if __name__ == '__main__':
    main()

    # import pickle
    # with open('optimization_info.pkl', 'rb') as f:
    #     optimization_info = pickle.load(f)
    #
    # # optimization_info.append({
    # #     'state_parameter': state_parameter,
    # #     'simulated_value': optimal_protocol[state_parameter][-1],
    # #     'current_value': current_value,
    # #     'std_du': std_energy,
    # #     'bar_df': df,
    # #     'bar_ddf': ddf,
    # #     'simulated_energies': energies,
    # #     'reweighted_energies': reweighted_energies
    # # })
    #
    # for info in optimization_info:
    #     if info['state_parameter'] == 'lambda_electrostatics':
    #         print('{:.2f}->{:.2f}: std(du) {:.3f}, df {:.3f}, ddf {:.3f}'.format(info['simulated_value'], info['current_value'],
    #                                                                              info['std_du'], info['bar_df'], info['bar_ddf']))
