# bem2d_python

## What is bem2d_python?

bem2d_python is a python module for 2D Boundary elements. I will eventually use it with HTOOL (https://github.com/htool-ddm/htool), Pierre Marchand's implementation of heirarchical matrices. Right now, it is in an early state of development, and is mostly a source of procrastination.

## Planned features
* Piecewise linear geometry
* 0th-order discontinuous basis functions
* 1st-order linear continuous basis functions
* Galerkin and collocation discretizations
* Support for Laplace and Helmholtz equation (real and complex wavenumber)
* Example scripts to create H marices and solve them using gmres
* Incident field for rough surface scattering, developed by Eric Thorsos (https://doi.org/10.1121/1.396188)
