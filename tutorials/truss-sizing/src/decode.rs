//! Normalized mixed-variable decoding.

use std::error::Error;

use crate::catalogue::sections;
use crate::ground::{GroundStructure, Node};

/// Minimum active-member cardinality.
pub const MIN_ACTIVE: usize = 8;
/// Maximum active-member cardinality.
pub const MAX_ACTIVE: usize = 40;

/// One selected ground-structure member and catalogue section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActiveMember {
    /// Index into [`GroundStructure::members`].
    pub member_index: usize,
    /// Index into the twelve-row section catalogue.
    pub section_index: usize,
}

/// Authoritatively decoded truss.
#[derive(Clone, Debug)]
pub struct DecodedDesign {
    /// Moved node coordinates.
    pub nodes: Vec<Node>,
    /// Exactly decoded active members.
    pub active: Vec<ActiveMember>,
}

/// Decision dimension derived from the instance.
#[must_use]
pub fn dimension(ground: &GroundStructure) -> usize {
    1 + 2 * ground.members.len() + 2 * ground.movable_nodes.len()
}

fn linear_integer(value: f64, lower: usize, upper: usize) -> Option<usize> {
    if !value.is_finite() || lower > upper {
        return None;
    }
    let bins = upper - lower + 1;
    Some(lower + ((value.clamp(0.0, 1.0) * bins as f64).floor() as usize).min(bins - 1))
}

/// Decode normalized controls into exact topology, sections, and geometry.
pub fn decode(controls: &[f64], ground: &GroundStructure) -> Result<DecodedDesign, Box<dyn Error>> {
    if controls.len() != dimension(ground) {
        return Err(format!(
            "expected {} controls, received {}",
            dimension(ground),
            controls.len()
        )
        .into());
    }
    if controls.iter().any(|value| !value.is_finite()) {
        return Err("all controls must be finite".into());
    }
    let member_count = ground.members.len();
    let active_count = linear_integer(controls[0], MIN_ACTIVE, MAX_ACTIVE)
        .ok_or("invalid active-count coordinate")?;
    let mut ranked = (0..member_count).collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        controls[1 + *left]
            .total_cmp(&controls[1 + *right])
            .then_with(|| left.cmp(right))
    });
    ranked.truncate(active_count);
    ranked.sort_unstable();
    let catalogue_count = sections().len();
    let active = ranked
        .into_iter()
        .map(|member_index| {
            let section_index = linear_integer(
                controls[1 + member_count + member_index],
                0,
                catalogue_count - 1,
            )
            .expect("finite section control must decode");
            ActiveMember {
                member_index,
                section_index,
            }
        })
        .collect();
    let mut nodes = ground.nodes.clone();
    let offset_start = 1 + 2 * member_count;
    let dx_limit = 0.15 * ground.bay_m;
    let dy_limit = 0.15 * ground.level_m;
    for (offset, node_index) in ground.movable_nodes.iter().enumerate() {
        nodes[*node_index].x += (2.0 * controls[offset_start + 2 * offset] - 1.0) * dx_limit;
        nodes[*node_index].y += (2.0 * controls[offset_start + 2 * offset + 1] - 1.0) * dy_limit;
    }
    Ok(DecodedDesign { nodes, active })
}

/// Conservative triangulated baseline in normalized coordinates.
#[must_use]
pub fn baseline_controls(ground: &GroundStructure) -> Vec<f64> {
    let member_count = ground.members.len();
    let mut controls = vec![0.5; dimension(ground)];
    let active_bins = MAX_ACTIVE - MIN_ACTIVE + 1;
    controls[0] = (MAX_ACTIVE - MIN_ACTIVE) as f64 / (active_bins - 1) as f64;
    let baseline = ground.baseline_members();
    for member_index in 0..member_count {
        controls[1 + member_index] = if baseline.binary_search(&member_index).is_ok() {
            0.1 + member_index as f64 * 1.0e-8
        } else {
            0.9 + member_index as f64 * 1.0e-8
        };
        controls[1 + member_count + member_index] = 11.5 / 12.0;
    }
    controls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_decode_to_inclusive_integer_bounds() {
        assert_eq!(linear_integer(0.0, 8, 40), Some(8));
        assert_eq!(linear_integer(1.0, 8, 40), Some(40));
        assert_eq!(linear_integer(-1.0, 0, 11), Some(0));
        assert_eq!(linear_integer(2.0, 0, 11), Some(11));
        assert_eq!(linear_integer(f64::NAN, 0, 11), None);
    }

    #[test]
    fn baseline_has_exact_cardinality_and_fixed_load_nodes() {
        let ground = GroundStructure::reference();
        let decoded = decode(&baseline_controls(&ground), &ground).unwrap();
        assert_eq!(decoded.active.len(), MAX_ACTIVE);
        assert_eq!(
            decoded.nodes[ground.load_nodes[0]],
            ground.nodes[ground.load_nodes[0]]
        );
        assert_eq!(
            decoded.nodes[ground.load_nodes[1]],
            ground.nodes[ground.load_nodes[1]]
        );
        assert_eq!(
            decoded
                .active
                .iter()
                .map(|member| member.member_index)
                .collect::<Vec<_>>(),
            ground.baseline_members()
        );
    }

    #[test]
    fn ties_are_resolved_by_member_index() {
        let ground = GroundStructure::reference();
        let mut controls = vec![0.5; dimension(&ground)];
        controls[0] = 0.0;
        let decoded = decode(&controls, &ground).unwrap();
        assert_eq!(
            decoded
                .active
                .iter()
                .map(|member| member.member_index)
                .collect::<Vec<_>>(),
            (0..MIN_ACTIVE).collect::<Vec<_>>()
        );
    }

    #[test]
    fn non_finite_controls_are_rejected() {
        let ground = GroundStructure::reference();
        let mut controls = baseline_controls(&ground);
        controls[7] = f64::INFINITY;
        assert!(decode(&controls, &ground).is_err());
    }

    #[test]
    fn integer_bins_are_balanced_on_midpoint_grid() {
        let mut counts = [0_usize; 12];
        let samples = 1_000_000;
        for index in 0..samples {
            let value = (index as f64 + 0.5) / samples as f64;
            counts[linear_integer(value, 0, 11).unwrap()] += 1;
        }
        let minimum = counts.iter().copied().min().unwrap();
        let maximum = counts.iter().copied().max().unwrap();
        assert!(maximum - minimum <= 1);
    }
}
