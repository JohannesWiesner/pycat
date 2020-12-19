# pycat

Python adapations of CAT12's third module 'Estimate Total Intracranial Volume (TIV)' and fourth module 'Check Sample'.

For more information, see the homepage of [CAT12](http://www.neuro.uni-jena.de/cat/) and the documentation of [the third and fourth module in the CAT12 manual](http://www.neuro.uni-jena.de/cat12/CAT12-Manual.pdf).

## Functions

`pycat.add_cat12_measures`: Extract TIV values from every CAT12 produced .XML file.

`pycat.check_sample_homogeneity`: Check sample for outliers  using distance measurements.

## Dependencies

The `pycat` module depends on the `nisupply` module (https://github.com/JohannesWiesner/nisupply).<br />In order for `pycat` to work, the `nisupply` module must be saved in the same directory.

