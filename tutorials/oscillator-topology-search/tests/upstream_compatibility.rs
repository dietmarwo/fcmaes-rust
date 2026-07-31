//! Sparse mass-action compatibility gate against the released ReBop 0.9.7.

use rebop::gillespie::{Gillespie as LocalGillespie, Rate as LocalRate};
use rebop_upstream::gillespie::{Gillespie as UpstreamGillespie, Rate as UpstreamRate};

fn local_model(seed: u64) -> LocalGillespie {
    let mut model = LocalGillespie::new_with_seed([20, 0], true, seed);
    model.add_reaction(LocalRate::lma_sparse(1.5, []), [1, 0]);
    model.add_reaction(LocalRate::lma_sparse(0.4, [(0, 1)]), [-1, 1]);
    model.add_reaction(LocalRate::lma_sparse(0.15, [(1, 1)]), [0, -1]);
    model
}

fn upstream_model(seed: u64) -> UpstreamGillespie {
    let mut model = UpstreamGillespie::new_with_seed([20, 0], true, seed);
    model.add_reaction(UpstreamRate::lma_sparse(1.5, []), [1, 0]);
    model.add_reaction(UpstreamRate::lma_sparse(0.4, [(0, 1)]), [-1, 1]);
    model.add_reaction(UpstreamRate::lma_sparse(0.15, [(1, 1)]), [0, -1]);
    model
}

#[test]
fn sparse_mass_action_matches_upstream_0_9_7() {
    for seed in [1, 42, 12_345] {
        let mut local = local_model(seed);
        let mut upstream = upstream_model(seed);
        for checkpoint in [0.25, 1.0, 5.0, 20.0] {
            local.advance_until(checkpoint);
            upstream.advance_until(checkpoint);
            assert_eq!(local.get_time(), upstream.get_time(), "seed={seed}");
            for species in 0..2 {
                assert_eq!(
                    local.get_species(species),
                    upstream.get_species(species),
                    "seed={seed}, checkpoint={checkpoint}, species={species}"
                );
            }
        }
    }
}
