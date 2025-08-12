In this folder are a number of assignments that can be used to teach Computational Materials Science. Feel free to download and edit!

The brachistochrone and interatomic potentials scripts are all .py files, with associated PDFs to explain the assignments and give answers with analysis of the data. The polymer chain modelling assignment contains a number of questions all in one Jupyter notebook.

The interatomic potentials assignments are based largely on the textbook Introduction to Computational Materials Science
Fundamentals to Applications by Richard Lesar. For more content and practice questions I highly recommend that textbook. The theory behind the polymer modelling assignment is based in part on the textbook Soft Condensed Matter by Richard A. Jones, whilst the question itself I wrote whilst working as a Teaching Assistant.

Info on what a brachistochrone curve is can be found at: https://mathworld.wolfram.com/BrachistochroneProblem.html
The brachistochrone folder contains the source code for determining the brachistochrone between two points, as well as a PDF with a description of code and how to interpret the results.

The interatomic potentials folder contains a number of scripts with the following:

sintered_particles - a Monte Carlo simulation to ascertain the void fraction of a cube containing sintered particles, with a graphical representation of the void fraction from all three axes.

The 4 scripts below are used in the assignments described in the 'Argon molecular dynamics' PDF.
MD_liquid_argon - simulates molecular dynamics of liquid argon using the Lennard-Jones potential, analyses thermodynamic properties, computes radial distribution and velocity autocorrelation functions, and visualises the results.
MD_solid_argon - does the same for solid argon.
MD_timestep_optimisation - tests multiple time steps in a molecular dynamics simulation of argon to find the optimal reduced timestep that minimises energy fluctuations.
MDLJ - simulates a system of atoms interacting via the Lennard-Jones potential in 3D space using the Velocity Verlet algorithm.

The 3 scripts below are used in the assignment described in the 'Comparing interatomic potentials' PDF
potentials - calculates the interatomic potentials using Lennard-Jones, Mie and Morse equations
fcc_neighbours - defines a function that returns the number of atoms and their radial distances for 17 neighbour shells in a face-centred cubic crystal lattice.
potentials_comparison - compares the above potentials.