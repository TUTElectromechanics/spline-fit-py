# Model fitting codes for magnetostriction

This code seeks for the best scalar potential in the least-squares sense, given measurements of its partial derivatives. This ensures that the fields generated by the fitted model are compatible in the physical sense (i.e. they are generated by the same potential). The scalar potential is represented by B-splines, allowing a fit to measured data for any material.

The Python version of the scripts is in spirit very similar to the original MATLAB version. This is a low-level implementation using the ``bspline`` module for Python, because the otherwise promising [LSQUnivariateSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.html#scipy.interpolate.LSQUnivariateSpline) of [SciPy](https://scipy.org) is too automatic for this use case; it does not accommodate the scenario where data is available on the derivative (instead of the function value).

Also, this model needs to be fitted in several dimensions at once; however, this can be performed on a meshgrid, which lends itself to an outer product approach using 1D splines along each axis.

When used with real measurement data, it is expected that the number of spline knots on each axis should be made (substantially?) smaller than the number of measurements, in order to denoise the data, thus avoiding overfitting and improving robustness. However, the collocation sites should be kept at the locations of the measurements in order to compute the fitting error correctly.

Technical details are documented in code comments.

Tested in Python 2.7 and 3.4.


## Getting started

The roles of the scripts are as follows:

 - [``fit_2d.py``](fit_2d.py): spline-fit (*B_x, lambda_xx*) against (*H_x, sigma_xx*).
 - [``fit_3d.py``](fit_3d.py): spline-fit (*B_x, lambda_xx, lambda_xy*) against (*H_x, sigma_xx, sigma_xy*).
 - [``Bernard_energy3D.py``](Bernard_energy3D.py): using the model of Bernard et al., generate mock measurement data for testing the fitting procedure. (This is slow due to the cubature implementation used in ``util/bfunc.py``. Pre-generated data files are included.)
 - [``comparison.py``](comparison.py): perform some symbolic calculations that can be used for computing the invariants I4, I5, I6 (whence also u, v, w) from H and sigma. This script also demonstrates the API calls to evaluate the fitted model.

Low-level utilities are located in:

 - [``util/bfunc.py``](util/bfunc.py): given (H, sigma), compute (B, lambda) from the model of Daniel et al. This is used by ``Bernard_energy3D.py``.
 - [``util/index.py``](util/index.py): indexing tricks for global DOF numbering of an outer product basis in nD.
 - [``util/plot.py``](util/plot.py): 3D wireframe plotting helper for ``Matplotlib``. This mainly fixes the problem that by default, Axes3D tends to underestimate the screen estate it needs if the plot is rotated by the user.

Exact references to the papers by Bernard et al. and Daniel et al. can be found in the corresponding modules.

### Notes

The solver algorithm can be chosen by enabling exactly one of the lines at the beginning of ``main()`` of ``fit_2d.py`` and ``fit_3d.py``.

For the 2D case, all algorithms work, with "sparse" (LSQR) producing the best fit for the test data.

For the 3D case, only "sparse_qr_solve" (SuiteSparseQR solver via `sparseqr`) finds a decent fit, due to the bad numerical scaling of the problem matrix. This particular scaling is required for equilibrating the components of the residual. If one instead equilibrates the matrix row and column norms, the residual components will differ by 11 orders of magnitude, leading to a bad fit.

One could work around this problem by applying dynamic range compression to the load vector before fitting the model, but this would produce a fit for the dynamics-compressed data, which would require undoing the compression when evaluating the fitted model. However, there is no need for this approach, since "sparse_qr_solve" can solve the original system, avoiding the need for extra processing at model evaluation time.

### Performance

The algorithm "optimize" (trust region reflective, TRF) is **very** slow; with this algorithm, the fitting of the 3D case will take several hours of CPU time. TRF is provided as a validity test; the result of the fitting should be the same regardless of the algorithm used.

The algorithm "sparse_qr" (SuiteSparseQR decomposer via `sparseqr`) is somewhat slow; due to the explicit construction of the orthogonal matrix Q, it may take a few minutes of CPU time to complete. Note that the variant "sparse_qr_solve" does **not** construct Q, and is very fast.

All the other algorithms complete in a few seconds of CPU time (on an i7).


## Dependencies

 - [NumPy](http://www.numpy.org)
 - [SciPy](https://scipy.org)
 - [Matplotlib](http://matplotlib.org)
 - [SymPy](http://www.sympy.org)
   - needed for [``comparison.py``](comparison.py)
 - [bspline](https://github.com/johntfoster/bspline)
   - B-spline basis functions and utilities
 - [sparseqr](https://github.com/yig/PySPQR)
   - sparse QR solver from SuiteSparseQR
   - required for [``fit_3d.py``](fit_3d.py); also works well for [``fit_2d.py``](fit_2d.py)
 - [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse)
   - CHOLMOD solver from SuiteSparse
   - optional for [``fit_2d.py``](fit_2d.py) (can also be enabled in [``fit_3d.py``](fit_3d.py), but does not produce good results there)
 - [wlsqm](https://github.com/Technologicat/python-wlsqm)
   - module ``wlsqm.utils.lapackdrivers``, for exposing LAPACK's [``DGEEQU``](http://www.netlib.org/lapack/explore-3.1.1-html/dgeequ.f.html) to the Python level
   - required for [``fit_2d.py``](fit_2d.py)
   - requires [Cython](http://cython.org)

### IMPORTANT

 - For CHOLMOD, use the new ``sksparse`` from ``scikit-sparse``, not the old ``scikits.sparse``!
 - CHOLMOD and SuiteSparseQR require the [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) header files.
   - in Debian-based distros, install ``libsuitesparse-dev`` (and maybe ``libcholmod2.1.2``)
   - in Red Hat based distros, install ``suitesparse-devel``


## Installation

For the Python dependencies, it is recommended to use the latest versions from ``pip``, because the ``wlsqm`` component requires the ``cython_lapack`` module that was added to SciPy rather recently. Also, old (even rather recent) versions of Cython may crash when compiling the code.

In addition to the Python dependencies, BLAS and LAPACK are required. In Debian-based distros, BLAS is available in the packages ``libblas3 libblas-dev`` (reference BLAS) or ``libopenblas-base libopenblas-dev`` (OpenBLAS). LAPACK is available in the packages ``liblapack3 liblapack-dev``.

Install the dependencies:

```bash
pip install --user cython numpy scipy matplotlib sympy scikit-sparse wlsqm bspline sparseqr
```

Then, finally, clone this project from git.


## License

[BSD](LICENSE.md)

The subdirectory ``freya`` contains components from the FREYA solver; [BSD](freya/LICENSE.md) license.
