# Binding free energy of bromodomain ligand

## Input files
The original GROMACS input files were downloaded from https://www.zenodo.org/record/57131#.WUfkShPyugR on June 1, 2017.
More specifically, the 4 files in each `input/ligandX/` folder were taken from:

- `complex.gro`: `Aldeghi-et-al-chemical-science-2016/crystal/LigandX/complex/fep_coord.gro`
- `complex.top`: `Aldeghi-et-al-chemical-science-2016/crystal/LigandX/complex/fep_topol.top`
- `solvent.gro`: `Aldeghi-et-al-chemical-science-2016/crystal/LigandX/ligand/fep_coord.gro`
- `solvent.top`: `Aldeghi-et-al-chemical-science-2016/crystal/LigandX/ligand/fep_topol.top`

The `gromacs_include/amber99sb-ildn.ff` was downloaded from https://github.com/gromacs/gromacs/tree/master/share/top/amber99sb-ildn.ff.

## Choice of the molecules
The ligands have been chosen to maximize the dynamical range among those that showed good convergence in [1].

## Differences with original protocol
- We don't run an NVT equilibration before NPT.
- We use an ewald tolerance of 1e-5 instead of 1e-6.
- We use a single cutoff for both electrostatics and sterics set to 11A instead of 12A and 10A respectively that
were employed for the GROMACS simulations.

## References
- [1] Aldeghi, M., Heifetz, A., Bodkin, M.J., Knapp, S. and Biggin, P.C., 2016. Accurate calculation of the absolute
free energy of binding for drug molecules. _Chemical Science_, 7(1), pp.207-218.
