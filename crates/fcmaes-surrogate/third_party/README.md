# Third-party notices

`src/sobol_directions.rs` holds the direction numbers of the first 32
Sobol dimensions with 32 bits each. They are the Joe–Kuo
`new-joe-kuo-6.21201` numbers, initialized with SciPy 1.17.0's
`scipy.stats._sobol._initialize_v`. The unscrambled points they generate
equal SciPy's `qmc.Sobol(d=32, scramble=False, bits=32)` for the first
1024 points in every dimension.

- [JOE_KUO_LICENSE.txt](JOE_KUO_LICENSE.txt): the direction numbers.
- [SCIPY_LICENSE.txt](SCIPY_LICENSE.txt): SciPy, which distributes and
  initializes them.

No other SciPy code or binary is included.
