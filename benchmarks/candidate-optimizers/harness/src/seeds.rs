//! Seed derivation.
//!
//! Every run seed is a function of the root seed (1 in round 1, 2 in round
//! 2), the problem name, the protocol, the repetition and the restart index
//! only, so all solvers see the same seeds. The mix is a chain of SplitMix64
//! finalizers:
//!
//! ```text
//! h = splitmix64(root)
//! h = splitmix64(h ^ fnv1a64(problem))
//! h = splitmix64(h ^ protocol_tag)        // 'S' = 0x53, 'R' = 0x52, 'T' = 0x54
//! h = splitmix64(h ^ repetition)
//! h = splitmix64(h ^ restart)             // 0 for protocol S
//! ```
//!
//! `fnv1a64` is the 64-bit FNV-1a hash of the UTF-8 problem name.

/// Root seed declared in the protocol (round 1).
pub const ROOT_SEED: u64 = 1;

/// Root seed declared for round 2 (`ROUND2.md`, section 4).
pub const ROOT_SEED_ROUND2: u64 = 2;

/// SplitMix64 finalizer (Steele, Lea and Flood 2014): add the golden-ratio
/// increment, then two xor-shift-multiply rounds.
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 64-bit FNV-1a hash of a string.
pub fn fnv1a64(text: &str) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// Seed of one run: `(problem, protocol tag, repetition, restart)` mixed into
/// the round-1 root seed. `protocol` is the protocol letter (`'S'` or `'R'`).
pub fn run_seed(problem: &str, protocol: char, repetition: u32, restart: u32) -> u64 {
    run_seed_from(ROOT_SEED, problem, protocol, repetition, restart)
}

/// Seed of one run under root seed `root` (same mix as [`run_seed`]).
pub fn run_seed_from(
    root: u64,
    problem: &str,
    protocol: char,
    repetition: u32,
    restart: u32,
) -> u64 {
    let mut h = splitmix64(root);
    h = splitmix64(h ^ fnv1a64(problem));
    h = splitmix64(h ^ u64::from(protocol));
    h = splitmix64(h ^ u64::from(repetition));
    splitmix64(h ^ u64::from(restart))
}

/// Seed of the uniform sample used to measure a problem's evaluation time.
pub fn measure_seed(problem: &str) -> u64 {
    splitmix64(splitmix64(ROOT_SEED) ^ fnv1a64(problem) ^ fnv1a64("measure"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn splitmix_matches_reference_values() {
        // First outputs of the reference SplitMix64 generator seeded with 0:
        // state advances by the increment, the finalizer maps it.
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
        assert_eq!(splitmix64(0x9E37_79B9_7F4A_7C15), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn fnv_matches_reference_values() {
        assert_eq!(fnv1a64(""), 0xCBF2_9CE4_8422_2325);
        assert_eq!(fnv1a64("a"), 0xAF63_DC4C_8601_EC8C);
    }

    #[test]
    fn run_seeds_are_stable() {
        // Frozen values: a change here changes every campaign seed.
        let frozen = [
            run_seed("cassini1", 'S', 0, 0),
            run_seed("cassini1", 'R', 0, 3),
            run_seed("lj38", 'S', 19, 0),
        ];
        assert_eq!(frozen, FROZEN);
    }

    const FROZEN: [u64; 3] = [
        0xCC64_BB3F_7A02_A0C2,
        0x94AC_F2B9_CF81_86A7,
        0x09A9_B373_4F13_CC8A,
    ];

    #[test]
    fn root_seed_one_is_the_round_one_derivation() {
        assert_eq!(
            run_seed_from(1, "lj13", 'R', 4, 7),
            run_seed("lj13", 'R', 4, 7)
        );
        assert_ne!(
            run_seed_from(2, "lj13", 'T', 0, 0),
            run_seed_from(1, "lj13", 'T', 0, 0)
        );
    }

    #[test]
    fn run_seeds_are_distinct() {
        let mut seen = HashSet::new();
        for problem in ["cassini1", "gtoc1", "lj13", "lj38", "bandpass"] {
            for protocol in ['S', 'R'] {
                for repetition in 0..20 {
                    for restart in 0..8 {
                        assert!(seen.insert(run_seed(problem, protocol, repetition, restart)));
                    }
                }
            }
        }
    }
}
