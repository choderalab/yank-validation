import os
import copy
import scipy
import pickle
import numpy as np

import mdtraj
import pymbar
from simtk import openmm, unit as unit
import openmmtools as mmtools
from yank import analyze


def extract_trajectories(experiment_directory):
    state_id = 0
    replica_id = 0
    trajectory_file_name = 'trajectory_flatbottom_cb7-argon'

    # Paths.
    nc_file_path = experiment_directory + 'complex.nc'
    state_trajectory_file_path = trajectory_file_name + '_state{}.pdb'.format(state_id)
    replica_trajectory_file_path = trajectory_file_name + '_replica{}.pdb'.format(replica_id)

    print('Extracting {} ...'.format(state_trajectory_file_path))
    traj = analyze.extract_trajectory(nc_path=nc_file_path, state_index=0, keep_solvent=False, image_molecules=True)
    traj.save(state_trajectory_file_path)

    print('Extracting {} ...'.format(replica_trajectory_file_path))
    traj = analyze.extract_trajectory(nc_path=nc_file_path, replica_index=0, keep_solvent=False, image_molecules=True)
    traj.save(replica_trajectory_file_path)


def compute_restraint_distance(positions_group1, positions_group2, weights_group1, weights_group2):
    """Compute the distance between the centers of mass of the two groups.

    The two positions given must have the same units.

    Parameters
    ----------
    positions_group1 : numpy.array
        The positions of the particles in the first CustomCentroidBondForce group.
    positions_group2 : numpy.array
        The positions of the particles in the second CustomCentroidBondForce group.
    weights_group1 : list of float
        The mass of the particle in the first CustomCentroidBondForce group.
    weights_group2 : list of float
        The mass of the particles in the second CustomCentroidBondForce group.
    """
    assert len(positions_group1) == len(weights_group1)
    assert len(positions_group2) == len(weights_group2)
    # Compute center of mass for each group.
    com_group1 = np.average(positions_group1, axis=0, weights=weights_group1)
    com_group2 = np.average(positions_group2, axis=0, weights=weights_group2)
    # Compute distance between centers of mass.
    distance = np.linalg.norm(com_group1 - com_group2)
    return distance


def compute_ssc(restraint_force, distance_threshold, kT, energy_threshold=None):
    """Compute the standard state correction."""
    r_min = 0 * unit.nanometers
    r_max = distance_threshold

    # Create a System object containing two particles connected by the reference force
    system = openmm.System()
    system.addParticle(1.0 * unit.amu)
    system.addParticle(1.0 * unit.amu)
    force = copy.deepcopy(restraint_force)
    force.setGroupParameters(0, [0])
    force.setGroupParameters(1, [1])
    # Disable the PBC if on for this approximation of the analytical solution
    force.setUsesPeriodicBoundaryConditions(False)
    system.addForce(force)

    # Create a Reference context to evaluate energies on the CPU.
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    context = openmm.Context(system, integrator, platform)

    # Set default positions.
    positions = unit.Quantity(np.zeros([2,3]), unit.nanometers)
    context.setPositions(positions)

    # Create a function to compute integrand as a function of interparticle separation.
    beta = 1 / kT

    def integrand(r):
        """
        Parameters
        ----------
        r : float
            Inter-particle separation in nanometers

        Returns
        -------
        dI : float
           Contribution to integrand (in nm^2).

        """
        positions[1, 0] = r * unit.nanometers
        context.setPositions(positions)
        state = context.getState(getEnergy=True)
        potential = state.getPotentialEnergy()

        if energy_threshold is not None and potential > energy_threshold:
            return 0.0

        dI = 4.0 * np.pi * r**2 * np.exp(-beta * potential)
        return dI

    # Integrate shell volume.
    shell_volume, shell_volume_error = scipy.integrate.quad(lambda r: integrand(r), r_min / unit.nanometers,
                                                            r_max / unit.nanometers) * unit.nanometers**3

    # Compute standard-state volume for a single molecule in a box of
    # size (1 L) / (avogadros number). Should also generate constant V0.
    liter = 1000.0 * unit.centimeters**3  # one liter
    standard_state_volume = liter / (unit.AVOGADRO_CONSTANT_NA*unit.mole)  # standard state volume

    print('\tShell volume: {}, standard volume:{}'.format(shell_volume, standard_state_volume))

    # Compute standard state correction for releasing shell restraints into standard-state box (in units of kT).
    DeltaG = - np.log(standard_state_volume / shell_volume)

    # Return standard state correction (in kT).
    return DeltaG


def get_energies(analyzer):
    """Extract the energy matrix in kn form from the analyzer.

    Returns
    -------
    u_kn : numpy.array
        The energies to feed to MBAR.
    N_k : numpy.array
        The number of samples generated from state k.
    uncorrelated_iterations : list of int
        The uncorrelated iteration indices.
    """
    sampled_u_kln, unsampled_u_kln = analyzer.get_states_energies()
    _, _, n_correlated_iterations = sampled_u_kln.shape

    # Discard equilibration.
    analyzer._get_equilibration_data_auto(input_data=sampled_u_kln)
    number_equilibrated, g_t, Neff_max = analyzer._equilibration_data
    sampled_u_kln = analyze.remove_unequilibrated_data(sampled_u_kln, number_equilibrated, -1)
    unsampled_u_kln = analyze.remove_unequilibrated_data(unsampled_u_kln, number_equilibrated, -1)

    # decorrelate_data subsample the energies only based on g_t so both ends up with same indices.
    sampled_u_kln = analyze.subsample_data_along_axis(sampled_u_kln, g_t, -1)
    unsampled_u_kln = analyze.subsample_data_along_axis(unsampled_u_kln, g_t, -1)
    u_kln, N_k = analyzer._prepare_mbar_input_data(sampled_u_kln, unsampled_u_kln)

    # Convert energies to u_kn format.
    u_kn = pymbar.utils.kln_to_kn(u_kln, N_k)

    # Figure out which iterations to consider.
    equilibrium_iterations = np.array(range(number_equilibrated, n_correlated_iterations))
    uncorrelated_iterations_indices = pymbar.timeseries.subsampleCorrelatedData(equilibrium_iterations, g_t)
    uncorrelated_iterations = equilibrium_iterations[uncorrelated_iterations_indices]
    print('g_t={:.3f}, u_kln.shape={}, u_kn.shape={}, sum(N_k)={}, n_uncorrelated_iterations={}'.format(
        g_t, u_kln.shape, u_kn.shape, np.sum(N_k), len(uncorrelated_iterations)))

    return u_kn, N_k, uncorrelated_iterations


def get_restraint_force(system):
    """Extract the CustomCentroidBondForce from the given system."""
    for i, force in enumerate(system.getForces()):
        if isinstance(force, openmm.CustomCentroidBondForce):
            restraint_force = copy.deepcopy(force)
            break
    return restraint_force


def get_end_states(analyzer):
    """Return the two unsampled states and a reduced version of them containing only the restraint force."""
    unsampled_states = analyzer._reporter.read_thermodynamic_states()[1]

    # Isolate CustomCentroidBondForce.
    restraint_force = get_restraint_force(unsampled_states[0].system)
    bond_parameters = restraint_force.getBondParameters(0)[1]
    try:  # FlatBottom
        print('Bond parameters: K={}, r0={}'.format(*bond_parameters))
    except IndexError:  # Harmonic
        print('Bond parameters: K={}'.format(*bond_parameters))

    # Obtain restraint's particle indices to compute restraint distance.
    particle_indices_group1, weights_group1 = restraint_force.getGroupParameters(0)
    particle_indices_group2, weights_group2 = restraint_force.getGroupParameters(1)
    assert len(weights_group1) == 0  # Use masses to compute centroid.
    assert len(weights_group2) == 0  # Use masses to compute centroid.

    # Convert tuples of np.integers to lists of ints.
    particle_indices_group1 = [int(i) for i in sorted(particle_indices_group1)]
    particle_indices_group2 = [int(i) for i in sorted(particle_indices_group2)]

    # Create new system with only solute and restraint forces.
    fully_interacting_system = unsampled_states[0].system
    reduced_system = openmm.System()
    for particle_indices_group in [particle_indices_group1, particle_indices_group2]:
        for i in particle_indices_group:
            reduced_system.addParticle(fully_interacting_system.getParticleMass(i))

    # Compute weights restrained particles.
    weights_group1 = [fully_interacting_system.getParticleMass(i) for i in particle_indices_group1]
    weights_group2 = [fully_interacting_system.getParticleMass(i) for i in particle_indices_group2]

    # Adapt CustomCentroidBondForce groups to reduced system.
    assert max(particle_indices_group1) < min(particle_indices_group2)
    n_atoms_group1 = len(particle_indices_group1)
    tot_n_atoms = n_atoms_group1 + len(particle_indices_group2)
    restraint_force = copy.deepcopy(restraint_force)
    restraint_force.setGroupParameters(0, list(range(n_atoms_group1)))
    restraint_force.setGroupParameters(1, list(range(n_atoms_group1, tot_n_atoms)))
    reduced_system.addForce(restraint_force)

    return (unsampled_states, reduced_system, particle_indices_group1,
            particle_indices_group2, weights_group1, weights_group2)


def get_state_indices_kn(analyzer, uncorrelated_iterations):
    """Return the replica state indices in kn format."""
    print('Loading replica state indices...')
    replica_state_indices = analyzer._reporter.read_replica_thermodynamic_states()
    n_correlated_iterations, n_replicas = replica_state_indices.shape
    print('n_correlated_iterations: {}, n_replicas: {}'.format(n_correlated_iterations, n_replicas))

    # Initialize output array.
    n_frames = n_replicas * len(uncorrelated_iterations)
    state_indices_kn = np.zeros(n_frames, dtype=np.int32)

    # Map kn columns to the sta
    for iteration_idx, iteration in enumerate(uncorrelated_iterations):
        for replica_idx, state_idx in enumerate(replica_state_indices):
            # Deconvolute index.
            state_idx = replica_state_indices[iteration, replica_idx]
            frame_idx = state_idx*len(uncorrelated_iterations) + iteration_idx
            # Set output array.
            state_indices_kn[frame_idx] = state_idx


def get_restrain_distances(analyzer, uncorrelated_iterations,
                           particle_indices_group1, particle_indices_group2,
                           weights_group1, weights_group2):
    """Compute the restrain distances for the given iterations.

    Parameters
    ----------
    analyzer : yank.analyze.ReplicaExchangeAnalyzer
    uncorrelated_iterations : list of int
        The uncorrelated iteration indices. This is needed to extract the
        correct positions/box vectors from the netcdf file.
    particle_indices_group1 : list of int
        The particle indices of the first CustomCentroidBondForce group.
    particle_indices_group2 : list of int
        The particle indices of the second CustomCentroidBondForce group.
    weights_group1 : list of float
        The mass of the particle in the first CustomCentroidBondForce group.
    weights_group2 : list of float
        The mass of the particles in the second CustomCentroidBondForce group.

    Returns
    -------
    restrain_distances_kn : np.array
        The restrain distances.
    state_indices_kn : list of int
        The index of the ThermodynamicState that generated this sample.
    """
    subset_particles_indices = list(analyzer._reporter.analysis_particle_indices)
    n_atoms = len(subset_particles_indices)

    print('Loading replica state indices...')
    replica_state_indices = analyzer._reporter.read_replica_thermodynamic_states()
    n_correlated_iterations, n_replicas = replica_state_indices.shape
    print('n_correlated_iterations: {}, n_replicas: {}'.format(n_correlated_iterations, n_replicas))

    # Create topology of the restrained atoms.
    print('Creating topology...')
    metadata = analyzer._reporter.read_dict('metadata')
    topology = mmtools.utils.deserialize(metadata['topography']).topology
    topology = topology.subset(subset_particles_indices)

    # Create output arrays. We unfold the replicas the same way
    # it is done during the kln_to_kn conversion.
    n_frames = n_replicas * len(uncorrelated_iterations)
    distances_kn = np.zeros(n_frames, dtype=np.float32)
    state_indices_kn = np.zeros(n_frames, dtype=np.int32)

    # Initialize trajectory object needed for imaging molecules.
    trajectory = mdtraj.Trajectory(xyz=np.zeros((n_atoms, 3)), topology=topology)

    # Pre-computing distances.
    print('Computing centroid distances...')
    for iteration_idx, iteration in enumerate(uncorrelated_iterations):
        print('\r\tProcessing iteration {}/{}'.format(iteration, n_correlated_iterations), end='')

        # Obtain solute only sampler states.
        sampler_states = analyzer._reporter.read_sampler_states(iteration=iteration,
                                                                analysis_particles_only=True)

        for replica_idx, sampler_state in enumerate(sampler_states):
            # Update trajectory positions/box vectors.
            trajectory.xyz = (sampler_state.positions[subset_particles_indices] / unit.nanometers).astype(np.float32)
            trajectory.unitcell_vectors = np.array([sampler_state.box_vectors / unit.nanometers], dtype=np.float32)
            trajectory.image_molecules(inplace=True)
            positions_group1 = trajectory.xyz[0][particle_indices_group1]
            positions_group2 = trajectory.xyz[0][particle_indices_group2]

            # Deconvolute index.
            state_idx = replica_state_indices[iteration, replica_idx]
            frame_idx = state_idx*len(uncorrelated_iterations) + iteration_idx

            # Set output arrays.
            state_indices_kn[frame_idx] = state_idx
            distances_kn[frame_idx] = compute_restraint_distance(positions_group1, positions_group2,
                                                                 weights_group1, weights_group2)
    print()

    # Set MDTraj units to distances.
    distances_kn = distances_kn * unit.nanometer

    return distances_kn, state_indices_kn


def verify_energies(analyzer, uncorrelated_iterations, end_states, u_kn, u_kln):
    """Recompute and verify the energies of the u_kn matrix.

    Parameters
    ----------
    analyzer : yank.analyze.ReplicaExchangeAnalyzer
    uncorrelated_iterations : list of int
        The uncorrelated iteration indices. This is needed to extract the
        correct positions/box vectors from the netcdf file.
    """
    print('Loading replica state indices...')
    replica_state_indices = analyzer._reporter.read_replica_thermodynamic_states()
    n_correlated_iterations, n_replicas = replica_state_indices.shape
    n_frames = n_replicas * len(uncorrelated_iterations)
    print('n_correlated_iterations: {}, n_replicas: {}'.format(n_correlated_iterations, n_replicas))

    print('Creating contexts for energy verification...')
    contexts_end_states = [None, None]

    platform = openmm.Platform.getPlatformByName('CUDA')
    platform.setPropertyDefaultValue('Precision', 'mixed')
    for state_idx, state in enumerate(end_states):
        integrator = openmm.VerletIntegrator(1.0*unit.femtoseconds)
        contexts_end_states[state_idx] = state.create_context(integrator, platform)

    # Obtain reference.
    _, energy_unsampled_states = analyzer._reporter.read_energies()
    energy_unsampled_states = energy_unsampled_states[uncorrelated_iterations]
    dtype = energy_unsampled_states.dtype

    # We compute the energies in different shapes to verify that the indexing is correct.
    u_kn_computed = np.zeros(shape=(2, n_frames), dtype=dtype)
    u_kln_computed = np.zeros(shape=(n_replicas+2, 2, len(uncorrelated_iterations)), dtype=dtype)
    energy_unsampled_states_computed = np.zeros(shape=energy_unsampled_states.shape, dtype=dtype)

    # Pre-computing distances.
    print('Computing energies...')
    for iteration_idx, iteration in enumerate(uncorrelated_iterations):
        print('\r\tProcessing iteration {}/{}'.format(iteration, n_correlated_iterations), end='')

        # Obtain full sampler states for energy verification if this is a checkpoint iteration.
        sampler_states = analyzer._reporter.read_sampler_states(iteration=iteration,
                                                                     analysis_particles_only=False)
        if sampler_states is None:
            continue

        for replica_idx, sampler_state in enumerate(sampler_states):
            # Deconvolute index.
            state_idx = replica_state_indices[iteration, replica_idx]
            frame_idx = state_idx*len(uncorrelated_iterations) + iteration_idx

            # Compute energies if this is a checkpoint iteration.
            for context_idx, (context, state) in enumerate(zip(contexts_end_states, end_states)):
                context.setPeriodicBoxVectors(*sampler_state.box_vectors)
                context.setPositions(sampler_state.positions)
                reduced_potential = state.reduced_potential(context)
                u_kn_computed[context_idx, frame_idx] = reduced_potential
                u_kln_computed[state_idx+1, context_idx, iteration_idx] = reduced_potential
                energy_unsampled_states_computed[iteration_idx, replica_idx, context_idx] = reduced_potential
    print()

    # First find all non-zero energies that correspond to checkpoint iterations.
    u_kn_checkpoint_frames = np.where(u_kn_computed[0] != 0.0)[0]
    u_kln_checkpoint_frames = np.where(u_kln_computed[1, 0, :])[0]
    energy_unsampled_states_checkpoint_frames = np.where(energy_unsampled_states_computed[:, 0, 0])[0]
    assert len(u_kn_checkpoint_frames) == len(u_kln_checkpoint_frames) * n_replicas
    assert len(u_kn_checkpoint_frames) == len(energy_unsampled_states_checkpoint_frames) * n_replicas

    # Verify energies. We need to keep the relative tolerance high
    # because the samples in state 0 with very high energy are very
    # much affected by numerical noise.
    assert np.allclose(energy_unsampled_states[energy_unsampled_states_checkpoint_frames],
                       energy_unsampled_states_computed[energy_unsampled_states_checkpoint_frames], rtol=1e-3)
    for context_idx in [0, -1]:
        assert np.allclose(u_kln_computed[:, context_idx, u_kln_checkpoint_frames],
                           u_kln[:, context_idx, u_kln_checkpoint_frames], rtol=1e-3)
        assert np.allclose(u_kn_computed[context_idx, u_kn_checkpoint_frames],
                           u_kn[context_idx, u_kn_checkpoint_frames], rtol=1e-3)


def update_energy_distribution_plot(axes, u_kn, plot_energy_distribution_states, u_min, distance_cutoff):
    for plot_state_idx, plot_state in enumerate(plot_energy_distribution_states):
        distribution = np.log(u_kn[plot_state] - u_min + 0.1)
        axes[plot_state_idx].hist(distribution, bins=500)
        axes[plot_state_idx].set_title('State index {}, cutoff {}'.format(plot_state, distance_cutoff))
        axes[plot_state_idx].set_xlabel('Reduced potential [log(kT)]')


def test_volume_reduction(experiment_directory, all_distance_cutoffs,
                          solvent_phase_free_energy, solvent_phase_error,
                          plot_energy_distribution_states=(),
                          save_lambda0_trajectory=False):
    """Find free energy after discarding samples outside the cutoff.

    Parameters
    ----------
    experiment_directory : str
        The path to the directory containing the YANK data for the experiment.
    all_distance_cutoffs : simtk.unit.Quantity array
        The cutoff distances to test (nits of distance).
    solvent_phase_free_energy : float
        The solvent phase free energy in kT. This is used just to display the
        final free energy, not only the complex one.
    solvent_phase_error : flaot
        The solvent phase free energy error in kT. This is used just to display
        the final free energy, not only the complex one.
    """
    complex_nc_path = experiment_directory + 'complex.nc'
    analyzer = analyze.get_analyzer(complex_nc_path)

    # Reproduce analysis here.
    print('Obtaining energies...')
    u_kn, N_k, uncorrelated_iterations = get_energies(analyzer)

    # Create figure if we require energy distribution.
    if len(plot_energy_distribution_states) > 0:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt

        nrows = len(all_distance_cutoffs)+1
        fig, axes = plt.subplots(nrows=nrows, ncols=len(plot_energy_distribution_states),
                                 figsize=(13, 4*nrows))
        u_min = np.min(u_kn)

        # Plot full data.
        update_energy_distribution_plot(axes[0], u_kn, plot_energy_distribution_states, u_min, 'none')

    # Load any restrained thermodynamic state.
    print('Creating system...')
    (end_states, reduced_system, particle_indices_group1,
     particle_indices_group2, weights_group1, weights_group2) = get_end_states(analyzer)
    restraint_force = get_restraint_force(reduced_system)
    kT = end_states[0].kT
    kT_to_kcalmol = kT / unit.kilocalories_per_mole

    # Verify that energy indexing is correct.
    # verify_energies(analyzer, uncorrelated_iterations, end_states, u_kn)
    del end_states

    # Compute restrain distances (or load them from cache).
    cache_file_path = 'cache_distances.p'
    if os.path.isfile(cache_file_path):
        print('Loading distances and state indices from {}...'.format(cache_file_path))
        with open(cache_file_path, 'rb') as f:
            distances_kn, state_indices_kn = pickle.load(f)
    else:
        distances_kn, state_indices_kn = get_restrain_distances(analyzer, uncorrelated_iterations,
                                                                particle_indices_group1, particle_indices_group2,
                                                                weights_group1, weights_group2)
        # Cache distances for future.
        with open(cache_file_path, 'wb') as f:
            pickle.dump((distances_kn, state_indices_kn), f)
    assert len(distances_kn) == u_kn.shape[1], 'Cache out-of-date. Delete and restart.'

    # Save the lambda0 trajectory to verify that the cutoff is correctly applied.
    if save_lambda0_trajectory:
        print('Extracting lambda0 trajectory...')
        n_replicas = len(u_kn) - 2
        lambda0_state_idx = n_replicas - 1
        lambda0_trajectory = analyze.extract_trajectory(nc_path=complex_nc_path, state_index=lambda0_state_idx,
                                                        discard_equilibration=False, keep_solvent=False,
                                                        image_molecules=True)
        # Take into account that the first sample in extract_trajectory is discarded.
        lambda0_trajectory = lambda0_trajectory[uncorrelated_iterations - 1]
        assert len(lambda0_trajectory) == len(uncorrelated_iterations)
        lambda0_iterations_kn = np.array([i for i in range(len(lambda0_trajectory)) for _ in range(n_replicas)])
        assert len(lambda0_iterations_kn) == len(distances_kn)

    # Determine samples outside the cutoff.
    print('Discarding samples outside the cutoff...')
    all_free_energies = np.zeros((len(all_distance_cutoffs), 4))
    for distance_cutoff_idx, distance_cutoff in enumerate(all_distance_cutoffs):
        print('Processing distance cutoff {}'.format(distance_cutoff))
        columns_to_keep = []

        for iteration_kn, distance in enumerate(distances_kn):
            if iteration_kn % 1000 == 0:
                print('\r\tProcessing kn iteration {}/{}'.format(iteration_kn, len(distances_kn)), end='')

            if distance > distance_cutoff:
                # Update the number of samples generated from its state. The +1
                # is necessary to take into account the first unsampled state.
                state_idx = state_indices_kn[iteration_kn]
                N_k[state_idx + 1] -= 1
            else:
                columns_to_keep.append(iteration_kn)
        print()

        # Drop all columns for next cutoff.
        u_kn = u_kn[:, columns_to_keep]
        distances_kn = distances_kn[columns_to_keep]
        state_indices_kn = state_indices_kn[columns_to_keep]

        # Save lambda0 trajectory with frames discarded.
        if save_lambda0_trajectory:
            lambda0_iterations_kn = lambda0_iterations_kn[columns_to_keep]
            frames_to_keep = []
            for column_idx, lambda0_iteration_kn in enumerate(lambda0_iterations_kn):
                if state_indices_kn[column_idx] == lambda0_state_idx:
                    frames_to_keep.append(lambda0_iteration_kn)
            print('\tSaving {} frames of the lambda0 trajectory...'.format(len(frames_to_keep)))
            lambda0_trajectory[frames_to_keep].save('traj{}A.pdb'.format(int(distance_cutoff / unit.angstroms)))

        # Update plot with this cutoff histogram.
        if len(plot_energy_distribution_states) > 0:
            update_energy_distribution_plot(axes[distance_cutoff_idx+1], u_kn,
                                            plot_energy_distribution_states, u_min,
                                            distance_cutoff)

        print('\tN_k: {}'.format(N_k))
        print('\tfinal u_kn.shape: {}, sum(N_k): {}'.format(u_kn.shape, np.sum(N_k)))

        # Compute free energy.
        print('\tComputing free energy...')
        mbar = pymbar.MBAR(u_kn, N_k)
        Deltaf_ij, dDeltaf_ij, theta_ij = mbar.getFreeEnergyDifferences()

        # Compute new standard state correction.
        print('\tComputing standard state correction...')
        ssc = compute_ssc(restraint_force, distance_cutoff, kT)

        # Print everything.
        print('\tSSC: {:.3f} kT'.format(ssc))
        print('\tComplex: {:.3f} +- {:.3f} kT'.format(Deltaf_ij[0, -1], dDeltaf_ij[0, -1]))
        free_energy = (solvent_phase_free_energy - ssc - Deltaf_ij[0, -1])
        error = np.sqrt(solvent_phase_error**2 + dDeltaf_ij[0, -1]**2)
        print('\tFree energy: {:.3f} +- {:.3f} kT ({:.3f} +- {:.3f} kcal/mol)'.format(free_energy, error,
                                                                                      free_energy * kT_to_kcalmol,
                                                                                      error * kT_to_kcalmol))
        all_free_energies[distance_cutoff_idx] = np.array([free_energy, error, Deltaf_ij[0, -1], ssc])

    # Save distribution figures.
    if len(plot_energy_distribution_states) > 0:
        plt.tight_layout()
        plt.savefig('reduced_potential_distribution.pdf')

    print('\n\n')
    print('Distance cutoffs')
    print(all_distance_cutoffs / unit.angstroms)
    print('Total free energies')
    print(all_free_energies[:, 0])
    print(all_free_energies[:, 0] * kT_to_kcalmol)
    print('Errors')
    print(all_free_energies[:, 1])
    print(all_free_energies[:, 1] * kT_to_kcalmol)
    print('Free energy complex phases')
    print(all_free_energies[:, 2])
    print(all_free_energies[:, 2] * kT_to_kcalmol)
    print('SSCs')
    print(all_free_energies[:, 3])
    print(all_free_energies[:, 3] * kT_to_kcalmol)

    print('\n\n')
    print(all_distance_cutoffs)
    print()
    print(all_free_energies.tolist())


def plot_simulation(all_distance_cutoffs, all_free_energies, kT_to_kcalmol, apr_value, exp_value, title):
    from matplotlib import pyplot as plt

    tot_free_energies = all_free_energies[:, 0]
    errors = all_free_energies[:, 1]
    complex_free_energies = all_free_energies[:, 2]
    sscs = all_free_energies[:, 3]
    all_distance_cutoffs = all_distance_cutoffs / unit.angstroms

    plt.figure(figsize=(12, 7))
    plt.errorbar(all_distance_cutoffs, tot_free_energies*kT_to_kcalmol, yerr=errors*kT_to_kcalmol,
                 label='complex + solvent + SSC')
    plt.plot(all_distance_cutoffs, -sscs*kT_to_kcalmol, label='SSC')
    plt.plot(all_distance_cutoffs, -complex_free_energies*kT_to_kcalmol, label='complex')
    # plt.plot(all_distance_cutoffs, [apr_value for _ in all_distance_cutoffs], label='APR')
    # plt.plot(all_distance_cutoffs, [exp_value for _ in all_distance_cutoffs], label='experiment')
    plt.legend()
    plt.xlabel('well radius (A)')
    plt.ylabel('$\Delta G$ (kcal/mol)')
    plt.title(title)
    plt.grid()
    # plt.show()
    plt.savefig('radius_dependency.png')


def plot_simulation_comparison(all_distance_cutoffs, *free_energies):
    from matplotlib import pyplot as plt
    plt.plot(all_distance_cutoffs, free_energies[0], label='simulation well_radius = 8A')
    plt.plot(all_distance_cutoffs, free_energies[1], label='simulation well_radius = 10A')
    plt.legend()
    plt.xlabel('well radius cutoff (A)')
    plt.ylabel('$\Delta G$ (kcal/mol)')
    plt.grid()
    plt.show()
    # plt.savefig('8A_vs_10A.pdf')


if __name__ == '__main__':
    # experiment_directory = '../cucurbit7uril/experiments/FlatBottom_cbguesta1/'
    # all_distance_cutoffs = [15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0] * unit.angstroms
    # experiment_directory = '../cucurbit7uril/experiments/Harmonic_cbguesta1/'
    # all_distance_cutoffs = [4.0, 3.0, 2.5, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6] * unit.angstroms
    # solvent_phase_free_energy = 36.036  # in kT
    # solvent_phase_error = 0.057  # in kT

    experiment_directory = '../cucurbit7uril/experiments_testcase/'
    all_distance_cutoffs = [12.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0] * unit.angstroms
    solvent_phase_free_energy = -2.967  # in kT
    solvent_phase_error = 0.026  # in kT

    # experiment_directory = '../cyclodextrin/experiments/experiment-flatbottom-bcd/100angstrom/'
    # experiment_directory = '../cyclodextrin/experiments/experiment-flatbottom-bcd/80angstrom/'
    # all_distance_cutoffs = [15.0, 12.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0] * unit.angstroms
    # solvent_phase_free_energy = 21.881  # in kT
    # solvent_phase_error = 0.050  # in kT

    # test_volume_reduction(experiment_directory, all_distance_cutoffs,
    #                       solvent_phase_free_energy, solvent_phase_error)

    # extract_trajectories(experiment_directory=experiment_directory)

    # CB7-A1 FlatBottom
    # all_distance_cutoffs = np.array([15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]) * unit.angstroms
    # all_free_energies = np.array([[-43.050818014675485, 0.07858094154532165, 76.94515284992772, 2.14166516474776],
    #                               [-43.05482279756996, 0.07931641128741458, 77.15613624728306, 1.9346865502869064],
    #                               [-43.06306790093634, 0.080263336698403, 77.3867052671106, 1.7123626338257405],
    #                               [-43.04423429539779, 0.08150950578919625, 77.60799978459266, 1.4722345108051316],
    #                               [-43.020568846016545, 0.08320904049731241, 77.8453684661803, 1.211200379836242],
    #                               [-43.00366611620156, 0.08630088943808825, 78.11439627577829, 0.9252698404232673],
    #                               [-43.03748918907286, 0.09337797678207478, 78.46430089562307, 0.6091882934497885],
    #                               [-43.002056153525, 0.10546755612327462, 78.78221696704436, 0.2558391864806379],
    #                               [-43.043519313760186, 0.11716128430987434, 79.22427430515312, -0.14475499139292952],
    #                               [-43.000333278660136, 0.1306519720562762, 79.64354030953484, -0.6072070308747044],
    #                               [-42.98341536642747, 0.1421456504740004, 80.17358706768404, -1.1541717012565686],
    #                               [-43.00230026444255, 0.145543676583743, 80.86190261964175, -1.823602355199198],
    #                               [-42.970447254678824, 0.14084686953462916, 81.69309582723336, -2.68664857255454]])
    # plot_simulation(all_distance_cutoffs, all_free_energies, 0.5924856518389833, -23.74, -14.1, 'CB7-A1 FlatBottom')

    # CB7-A1 Harmonic
    # all_distance_cutoffs = [4.0, 3.0, 2.5, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6] * unit.angstroms
    # all_free_energies = np.array([[-42.6597155699115, 0.0722251064961514, 84.25330704156971, -5.5575914716582115],
    #                               [-42.65465979870171, 0.07222602418949495, 84.2491911731425, -5.558531374440799],
    #                               [-42.58867604796444, 0.07224027891833705, 84.19213240057994, -5.567456352615498],
    #                               [-42.08462588515295, 0.07233897760323635, 83.74362595709496, -5.623000071942011],
    #                               [-41.48460250883656, 0.07245120574606577, 83.20204516827437, -5.681442659437807],
    #                               [-40.338388906450014, 0.07267525125351228, 82.1528693617624, -5.778480455312381],
    #                               [-38.3564464692294, 0.07307178864081486, 80.32367290879823, -5.931226439568834],
    #                               [-34.94017471452812, 0.07375955267694677, 77.1379363493692, -6.161761634841077],
    #                               [-29.332802571701293, 0.07506936139018816, 71.86867512461494, -6.499872552913644],
    #                               [-19.061787922351193, 0.07824186145342905, 62.08760858265012, -6.989820660298926],
    #                               [2.8652283214332357, 0.08750185208008825, 40.87873888087725, -7.707967202310486]])
    # plot_all(all_distance_cutoffs, all_free_energies, 0.5924856518389833, -23.74, -14.1, 'CB7-A1 Harmonic')


    # CB7-A1 Harmonic
    all_distance_cutoffs = [12.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0] * unit.angstroms
    all_free_energies = np.array([[-2.226232649538564, 0.040983842863336416, -2.2130018612665676, 1.4722345108051316],
                                  [-2.018183426316675, 0.05116829280161977, -1.8740864141065927, 0.9252698404232673],
                                  [-1.939825294453693, 0.059524434436231774, -1.6363629989960957, 0.6091882934497885],
                                  [-1.791706558891311, 0.07902246613060583, -1.431132627589327, 0.2558391864806379],
                                  [-1.746699454285635, 0.09811304357714501, -1.0755455543214358, -0.14475499139292952],
                                  [-1.6646076133812389, 0.10810189642255208, -0.6951853557440568, -0.6072070308747044],
                                  [-1.6026851693399582, 0.11314573544271259, -0.21014312940347324, -1.1541717012565686]])
    plot_simulation(all_distance_cutoffs, all_free_energies, 0.5924856518389833, 0.0, 0.0, 'CB7-Argon FlatBottom')

    # BCD FlatBottom
    # all_distance_cutoffs = [15.0, 12.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0] * unit.angstroms
    # free_energies_8A = np.array([-3.6666397, -3.6666397, -3.6666397, -3.6666397, -2.12050783, 9.53672727, 21.94774679, 34.06508886])
    # free_energies_10A = np.array([-3.20420878, -3.20420878, -1.38738852, 19.17550023, 51.31424751, 86.72881766, 122.80667675, 154.13097142])
    # plot_simulation_comparison(all_distance_cutoffs/unit.angstroms, free_energies_8A, free_energies_10A)
