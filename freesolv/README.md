# Hydration free energy of the FreeSolv Database (reduced)

## Manifest

- `freesolv.yaml`: Setup and simulation protocol for YANK. The protocol is chosen to be as close as possible to the
one described in [1].
- `freesolv-mini.smiles`: Sub-set of the [FreeSolv database](https://github.com/MobleyLab/FreeSolv) containing SMILES,
name, experimental and computed hydration free energies of the molecules used for validation.
- `expected_output.yaml`: Contains the free energies and statistical errors that are expected. It is used by the code
for automatic analysis.
- `run-yank.sh`: Bash script to run YANK serially.
- `run-torque-yank.sh`: Torque script to run YANK with MPI on the cluster.
- `README.md`: This file.

## Choice of the molecules from FreeSolv

The 5 molecules in the `freesolv-mini.smiles` set were extracted from the full FreeSolv database. The molecules
were chosen firstly to span a large dynamic range and secondly to have reasonable agreement between experimental
and reference computed value.

## Differences in the protocol

There are few difference from the protocol adopted in [1]:
- Instead of running an NPT equilibration followed by an NVT production simulation, we run all in NPT and use
automatic equilibration detection in the analysis.
- Using tleap instead of packmol, we can't control the exact number of water molecules (1309 in the reference paper).
We instead add water to create a 13A buffer around the molecule.
- We use an ewald tolerance of 1e-5 instead of 1e-6.
- We use a single cutoff for both electrostatics and sterics set to 11A instead of 12A and 10A respectively that
were employed for the GROMACS simulations.
- We use hamiltonian replica exchange.

## References

- [1] Duarte Ramos Matos, G., Kyu, D.Y., Loeffler, H.H., Chodera, J.D., Shirts, M.R. and Mobley, D.L., 2017.
Approaches for Calculating Solvation Free Energies and Enthalpies Demonstrated with an Update of the FreeSolv Database.
Journal of Chemical & Engineering Data.
