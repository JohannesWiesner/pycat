# pycat

A python package that imitates functions from the [Computational Anatomy Toolbox - CAT12](http://www.neuro.uni-jena.de/cat/).
For more information, see the documentation of [the third and fourth module in the CAT12 manual](http://www.neuro.uni-jena.de/cat12/CAT12-Manual.pdf).

## Functions

`pycat.get_TIV()`: Extract TIV values from every .XML file that CAT12 produces. This function works as desired.

`pycat.check_sample_homogeneity`: Check sample for outliers  using distance measurements. This function currently does not provided the exact logic that CAT12 has. Have a look at the FIXME.

## Dependencies

The `pycat` module depends on the `nisupply` module (https://github.com/JohannesWiesner/nisupply). In order for `pycat` to work, the `nisupply` module must be saved in the same directory.

## Note
`pycat` is still under development so feel free to add more functionality to it by sending pull requests!