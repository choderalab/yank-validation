import copy
import scipy
import numpy as np

import mdtraj
import pymbar
from simtk import openmm, unit as unit
import openmmtools as mmtools
from yank import analyze


def extract_trajectories(experiment_directory):
    state_id = 0
    replica_id = 0
    trajectory_file_name = 'trajectory_flatbottom_cb7-a1'

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


def compute_restraint_distance(positions, n_atoms_group1, weights_group1, weights_group2):
    """Compute the distance between the centers of mass of the two groups."""
    assert len(weights_group1) == n_atoms_group1
    assert len(weights_group1) + len(weights_group2) == len(positions)
    # Compute center of mass for each group.
    com_group1 = np.average(positions[:n_atoms_group1], axis=0, weights=weights_group1)
    com_group2 = np.average(positions[n_atoms_group1:], axis=0, weights=weights_group2)
    # Compute distance between centers of mass.
    distance = np.linalg.norm(com_group1 - com_group2)
    return distance


def compute_ssc(restraint_force, distance_threshold, energy_threshold, kT):
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
    print('Shell volume: {}'.format(shell_volume))

    # Compute standard-state volume for a single molecule in a box of
    # size (1 L) / (avogadros number). Should also generate constant V0.
    liter = 1000.0 * unit.centimeters**3  # one liter
    standard_state_volume = liter / (unit.AVOGADRO_CONSTANT_NA*unit.mole)  # standard state volume

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
    equilibrium_iterations = range(number_equilibrated, n_correlated_iterations)
    uncorrelated_iterations = pymbar.timeseries.subsampleCorrelatedData(equilibrium_iterations, g_t)
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


def get_system(analyzer):
    """Build a system containing only the restraint and the restraint particles."""
    fully_interacting_state = analyzer._reporter.read_thermodynamic_states()[1][0]
    fully_interacting_system = fully_interacting_state.system
    kT = fully_interacting_state.kT

    # Isolate CustomCentroidBondForce.
    restraint_force = get_restraint_force(fully_interacting_system)
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
    system = openmm.System()
    for particle_indices_group in [particle_indices_group1, particle_indices_group2]:
        for i in particle_indices_group:
            system.addParticle(fully_interacting_system.getParticleMass(i))

    # Adapt CustomCentroidBondForce groups to reduced system.
    n_atoms_group1 = len(particle_indices_group1)
    tot_n_atoms = n_atoms_group1 + len(particle_indices_group2)
    restraint_force.setGroupParameters(0, list(range(n_atoms_group1)))
    restraint_force.setGroupParameters(1, list(range(n_atoms_group1, tot_n_atoms)))
    system.addForce(restraint_force)

    return system, kT, particle_indices_group1, particle_indices_group2


def get_trajectory(analyzer, uncorrelated_iterations, particle_indices_group1, particle_indices_group2):
    """Creates a trajectory with only the restrained atoms.

    Parameters
    ----------
    analyzer : yank.analyze.ReplicaExchangeAnalyzer
    uncorrelated_iterations : list of int
        The uncorrelated iteration indices. This is needed to extract the
        correct positions/box vectors from the netcdf file.
    particle_indices_group1 : list of int
        The particle indices of the first CustomCentroidBondForce group.
    particle_indices_group2 : list of int
        The particle indices of the first CustomCentroidBondForce group.

    Returns
    -------
    trajectory : mdtraj.Trajectory
        The trajectory of the restrained particles at the uncorrelated
        iterations.
    state_indices_kn : list of int
        The index of the ThermodynamicState that generated this sample.
    """
    restrained_particle_indices = particle_indices_group1 + particle_indices_group2

    print('Loading replica state indices...')
    replica_state_indices = analyzer._reporter.read_replica_thermodynamic_states()
    n_correlated_iterations, n_replicas = replica_state_indices.shape
    print('n_correlated_iterations: {}, n_replicas: {}'.format(n_correlated_iterations, n_replicas))

    # Create topology of the restrained atoms.
    print('Creating topology...')
    metadata = analyzer._reporter.read_dict('metadata')
    topology = mmtools.utils.deserialize(metadata['topography']).topology
    topology = topology.subset(restrained_particle_indices)

    # Create positions and box vectors that we'll need to initialize a MDTraj Trajectory.
    print('Creating trajectory of restrained particles...')
    # We unfold the replicas the same way it is done during the kln_to_kn conversion.
    n_frames = len(uncorrelated_iterations) * n_replicas
    n_atoms = len(restrained_particle_indices)
    xyz = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    unitcell_vectors = np.zeros((n_frames, 3, 3), dtype=np.float32)
    state_indices_kn = np.zeros(n_frames, dtype=np.int32)

    for uncorrelated_iteration, correlated_iteration in enumerate(uncorrelated_iterations):
        print('\r\tProcessing iteration {}/{}'.format(correlated_iteration, n_correlated_iterations), end='')
        sampler_states = analyzer._reporter.read_sampler_states(iteration=correlated_iteration,
                                                                analysis_particles_only=True)
        for replica_idx, sampler_state in enumerate(sampler_states):
            state_idx = replica_state_indices[correlated_iteration, replica_idx]  # Deconvolute.
            frame_idx = uncorrelated_iteration*n_replicas + replica_idx
            xyz[frame_idx, :] = sampler_state.positions[restrained_particle_indices] / unit.nanometers
            unitcell_vectors[frame_idx, :] = sampler_state.box_vectors / unit.nanometers
            state_indices_kn[frame_idx] = state_idx
    print()

    # Build the trajectory.
    trajectory = mdtraj.Trajectory(xyz, topology)
    trajectory.unitcell_vectors = unitcell_vectors

    print('Imaging molecules...')
    atoms = list(topology.atoms)
    n_atoms_group1 = len(particle_indices_group1)
    anchor_molecules = [atoms[:n_atoms_group1]]
    other_molecules = [atoms[n_atoms_group1:]]
    trajectory.image_molecules(inplace=True, anchor_molecules=anchor_molecules,
                               other_molecules=other_molecules)

    return trajectory, state_indices_kn


def test_volume_reduction(experiment_directory, all_distance_cutoffs,
                          solvent_phase_free_energy, solvent_phase_error,
                          energy_threshold=None):
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

    # Load any restrained thermodynamic state.
    print('Creating system...')
    system, kT, particle_indices_group1, particle_indices_group2 = get_system(analyzer)
    restraint_force = get_restraint_force(system)
    kT_to_kcalmol = kT / unit.kilocalories_per_mole

    # Create the trajectory.
    trajectory, state_indices_kn = get_trajectory(analyzer, uncorrelated_iterations,
                                                  particle_indices_group1, particle_indices_group2)
    assert len(trajectory) == u_kn.shape[1]

    # Compute weights to compute the centers of mass.
    masses = np.array([a.element.mass for a in trajectory.top.atoms])
    weights_group1 = masses[:len(particle_indices_group1)]
    weights_group2 = masses[len(particle_indices_group1):]

    # Convert energy_threshold in kT and create context to compute energies.
    if energy_threshold is not None:
        energy_threshold *= kT
        print('Creating context...')
        platform = openmm.Platform.getPlatformByName('CPU')
        integrator = openmm.VerletIntegrator(1.0*unit.femtoseconds)
        context = openmm.Context(system, integrator, platform)

    # Determine samples outside the cutoff.
    print('Discarding samples outside the cutoff...')
    all_free_energies = np.zeros((len(all_distance_cutoffs), 4))
    for distance_cutoff_idx, distance_cutoff in enumerate(all_distance_cutoffs):
        print('Processing distance cutoff {}'.format(distance_cutoff))
        columns_to_keep = []

        for iteration_kn in range(len(trajectory)):
            print('\r\tProcessing kn iteration {}/{}'.format(iteration_kn, len(trajectory)), end='')
            discard = False
            positions = trajectory.xyz[iteration_kn]

            # Verify distance between centers of mass.
            distance = compute_restraint_distance(positions, len(particle_indices_group1),
                                                  weights_group1, weights_group2)
            distance *= unit.nanometer

            if distance > distance_cutoff:
                discard = True
            elif energy_threshold is not None:
                # Compute the energy of the restraint for the frame.
                context.setPositions(positions)
                context.setPeriodicBoxVectors(*trajectory.unitcell_vectors[iteration_kn])
                openmm_state = context.getState(getEnergy=True)
                potential_energy = openmm_state.getPotentialEnergy()

                if potential_energy > energy_threshold:
                    discard = True

            # Discard or keep sample.
            if discard:
                # Update the number of samples generated from its state. The +1
                # is necessary to take into account the first unsampled state.
                state_idx = state_indices_kn[iteration_kn]
                N_k[state_idx + 1] -= 1
            else:
                columns_to_keep.append(iteration_kn)
        print()

        # Drop all columns.
        u_kn = u_kn[:, columns_to_keep]
        trajectory = trajectory[columns_to_keep]
        state_indices_kn = state_indices_kn[columns_to_keep]

        print('\tN_k: {}'.format(N_k))
        print('\tfinal u_kn.shape: {}, sum(N_k): {}'.format(u_kn.shape, np.sum(N_k)))

        # Compute free energy.
        print('\tComputing free energy...')
        mbar = pymbar.MBAR(u_kn, N_k)
        Deltaf_ij, dDeltaf_ij, theta_ij = mbar.getFreeEnergyDifferences()

        # Compute new standard state correction.
        print('\tComputing standard state correction...')
        ssc = compute_ssc(restraint_force, distance_cutoff, energy_threshold, kT)

        # Print everything.
        print('\tSSC: {:.3f} kT'.format(ssc))
        print('\tComplex: {:.3f} +- {:.3f} kT'.format(Deltaf_ij[0, -1], dDeltaf_ij[0, -1]))
        free_energy = (solvent_phase_free_energy - ssc - Deltaf_ij[0, -1])
        error = np.sqrt(solvent_phase_error**2 + dDeltaf_ij[0, -1]**2)
        print('\tFree energy: {:.3f} +- {:.3f} kT ({:.3f} +- {:.3f} kcal/mol)'.format(free_energy, error,
                                                                                      free_energy * kT_to_kcalmol,
                                                                                      error * kT_to_kcalmol))
        all_free_energies[distance_cutoff_idx] = np.array([free_energy, error, Deltaf_ij[0, -1], ssc])

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
    plt.errorbar(all_distance_cutoffs, tot_free_energies*kT_to_kcalmol, yerr=errors*kT_to_kcalmol,
                 label='complex + solvent + SSC')
    plt.plot(all_distance_cutoffs, -sscs*kT_to_kcalmol, label='SSC')
    plt.plot(all_distance_cutoffs, -complex_free_energies*kT_to_kcalmol, label='complex')
    plt.plot(all_distance_cutoffs, [apr_value for _ in all_distance_cutoffs], label='APR')
    plt.plot(all_distance_cutoffs, [exp_value for _ in all_distance_cutoffs], label='experiment')
    plt.legend()
    plt.xlabel('well radius (A)')
    plt.ylabel('$\Delta G$ (kcal/mol)')
    plt.title(title)
    plt.grid()
    plt.show()
    # plt.savefig('radius_dependency.pdf')


def plot_simulation_comparison(*free_energies):
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
    experiment_directory = '../cucurbit7uril/experiments/FlatBottom_cbguesta1/'
    # experiment_directory = '../cucurbit7uril/experiments/Harmonic_cbguesta1/'
    all_distance_cutoffs = [15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0] * unit.angstroms
    # all_distance_cutoffs = [4.0, 3.0, 2.5, 2.0, 1.8, 1.6, 1.4] * unit.angstroms
    # all_distance_cutoffs = [2.0, 1.8, 1.6, 1.4] * unit.angstroms
    solvent_phase_free_energy = 36.036  # in kT
    solvent_phase_error = 0.057  # in kT

    # experiment_directory = '../cyclodextrin/experiments/experiment-flatbottom-bcd/100angstrom/'
    # experiment_directory = '../cyclodextrin/experiments/experiment-flatbottom-bcd/80angstrom/'
    # all_distance_cutoffs = [15.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0] * unit.angstroms
    # solvent_phase_free_energy = 21.881  # in kT
    # solvent_phase_error = 0.050  # in kT

    test_volume_reduction(experiment_directory, all_distance_cutoffs,
                          solvent_phase_free_energy, solvent_phase_error)

    # extract_trajectories(experiment_directory=experiment_directory)

    # # CB7-A1 FlatBottom
    # all_distance_cutoffs = np.array([15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]) * unit.angstroms
    # all_free_energies = np.array([[-41.29538565693429, 0.07892776776875059, 75.18972049227827, 2.1416651646560285],
    #                               [-36.986255811213944, 0.08035137281882047, 71.08756926144657, 1.934686549767373],
    #                               [-32.956812770994944, 0.08155130040007391, 67.28045013617987, 1.7123626348150778],
    #                               [-29.015041709433888, 0.08275852530497753, 63.57880719912282, 1.4722345103110697],
    #                               [-25.249379124108685, 0.08416152203272129, 60.07417874548465, 1.211200378624032],
    #                               [-21.328864309455724, 0.0859967573347175, 56.43959446903651, 0.9252698404192172],
    #                               [-16.911304544718732, 0.08903801525365707, 52.33811625500648, 0.6091882897122494],
    #                               [-13.984888873022363, 0.0928255380905386, 49.76504968674607, 0.25583918627629887],
    #                               [-11.972280846693806, 0.09994708835770875, 48.15303583860627, -0.14475499191246283],
    #                               [-10.316279982532535, 0.11403450933438981, 46.9594870139013, -0.6072070313687663],
    #                               [-8.58077569663105, 0.13559837453126697, 45.770947397064475, -1.1541717004334235],
    #                               [-6.5746707804458495, 0.16084109765434038, 44.43427313489757, -1.8236023544517257],
    #                               [-4.306424911018965, 0.18325628233478225, 43.02907348054873, -2.6866485695297677]])
    # plot_all(all_distance_cutoffs, all_free_energies, 0.5924856518389833, -23.74, -14.1, 'CB7-A1 FlatBottom')

    # # BCD FlatBottom
    # all_distance_cutoffs = np.array([15.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0]) * unit.angstroms
    # free_energies_8A = np.array([-3.6666397, -3.6666397, -3.6666397, -2.09261929, 9.5091256, 21.73835951, 33.77371354])
    # free_energies_10A = np.array([-3.20420878, -1.44881098, 19.15943377, 50.85284782, 86.23156173, 121.69365729, 152.5393971])
    # plot_simulation_comparison(free_energies_8A, free_energies_10A)