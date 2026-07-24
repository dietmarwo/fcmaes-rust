"""Native Rust implementation of fast, gradient-free optimization.

The documented names in this module form the public Python facade. The
low-level extension remains available as :data:`native` for signature
inspection and advanced use.
"""

from importlib.metadata import version as _distribution_version

from . import _fcmaes_ext as native

__version__ = _distribution_version("fcmaes-rust")

phase1_build_info = native.phase1_build_info

ACMA = native.ACMA
Archive = native.Archive
Bite = native.Bite
CRFMNES = native.CRFMNES
DE = native.DE
MODE = native.MODE
PGPE = native.PGPE

optimize_acma = native.optimize_acma
optimize_bite = native.optimize_bite
optimize_crfmnes = native.optimize_crfmnes
optimize_da = native.optimize_da
optimize_de = native.optimize_de
optimize_pgpe = native.optimize_pgpe

minimize_retry = native.minimize_retry
minimize_advanced_retry = native.minimize_advanced_retry
minimize_moretry = native.minimize_moretry

gtop_gtoc1 = native.gtop_gtoc1
gtop_cassini1 = native.gtop_cassini1
gtop_cassini1_minlp = native.gtop_cassini1_minlp
gtop_cassini2 = native.gtop_cassini2
gtop_cassini2_minlp = native.gtop_cassini2_minlp
gtop_messenger = native.gtop_messenger
gtop_messengerfull = native.gtop_messengerfull
gtop_rosetta = native.gtop_rosetta
gtop_sagas = native.gtop_sagas
gtop_tandem = native.gtop_tandem
gtop_tandem_unconstrained = native.gtop_tandem_unconstrained

__all__ = [
    "__version__",
    "ACMA",
    "Archive",
    "Bite",
    "CRFMNES",
    "DE",
    "MODE",
    "PGPE",
    "gtop_cassini1",
    "gtop_cassini1_minlp",
    "gtop_cassini2",
    "gtop_cassini2_minlp",
    "gtop_gtoc1",
    "gtop_messenger",
    "gtop_messengerfull",
    "gtop_rosetta",
    "gtop_sagas",
    "gtop_tandem",
    "gtop_tandem_unconstrained",
    "minimize_advanced_retry",
    "minimize_moretry",
    "minimize_retry",
    "native",
    "optimize_acma",
    "optimize_bite",
    "optimize_crfmnes",
    "optimize_da",
    "optimize_de",
    "optimize_pgpe",
    "phase1_build_info",
]
