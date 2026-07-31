# Attribution and scope

The topology grammar, additive Hill model, variable parameter layout, random
and `(1+1)` evolutionary controls, structural motif flags and split-brain
architecture are ported from
[`dietmarwo/autoresearch-circuit`](https://github.com/dietmarwo/autoresearch-circuit).
The Rust implementation is newly written for this tutorial.

The spectral, amplitude and autocorrelation scoring method is adapted from this
repository's [`rebop-oscillator`](../rebop-oscillator/) tutorial. It is applied
to a different three-species model and extended with broad-participation and
cross-gene period-coherence terms.

The provider-independent subprocess shape follows
[`gtoc1-route-search`](../gtoc1-route-search/), but the prompt, grammar and
response schema are domain-specific. Its checked mock contains no historical
or held-out topology.

The runtime Gillespie engine and expression tree under `vendor/rebop/` are a
narrow compatibility copy of ReBop 0.9.7 by Virgile Andreani, licensed MIT.
See [DEPENDENCY_NOTICE.md](DEPENDENCY_NOTICE.md).

The contribution of this tutorial is not a claim to have invented these
motifs or the split-brain idea. It is a native, tested and replayable
implementation which makes discrete proposal quality measurable against
separately optimized references.
