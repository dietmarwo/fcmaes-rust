# Held-out topology references

The reference set is structural. Only the repressilator is promised to be an
oscillator under this reduced model. Each row is optimized separately under
the same inner budget and is excluded from outer proposal history.

| Name | Canonical vector | Active signed core | Interpretation |
|---|---|---|---|
| repressilator | `000200220` | `A -| B -| C -| A` | oscillator reference |
| Goodwin-like | `000100120` | `A -> B -> C -| A` | signed negative-feedback core, not Goodwin model equivalence |
| positive cycle | `000100110` | `A -> B -> C -> A` | structural comparison; no oscillation promise |
| toggle control | `000212000` | `A -| B`, `B -| A`, plus `A -> C` | bistable negative control; third edge only prevents isolated C |

The repressilator reference follows Elowitz and Leibler's three-repressor
synthetic oscillator ([DOI 10.1038/35002125](https://doi.org/10.1038/35002125)).
The Goodwin label refers only to the signed negative-feedback idea in Goodwin's
enzymatic-control oscillator
([DOI 10.1016/0065-2571(65)90067-1](https://doi.org/10.1016/0065-2571(65)90067-1)).
The mutual-inhibition control follows the genetic toggle switch of Gardner,
Cantor and Collins
([DOI 10.1038/35002131](https://doi.org/10.1038/35002131)).

## Classifier

Both directed three-cycles (`A→B→C→A` and `A→C→B→A`) are inspected:

- `(2,2,2)` → `repressilator`;
- any permutation of `(1,1,2)` → `goodwin-like`;
- `(1,1,1)` → `positive-cycle`;
- another fully active signed cycle → `mixed-cycle`; and
- any mutually inhibiting pair → `toggle-core`.

An outer arm “exactly rediscovers” a reference only when all nine canonical
slots match. Finding the same motif class in another orientation is recorded
as class discovery, not exact rediscovery.
