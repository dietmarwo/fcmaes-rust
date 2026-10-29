//! Recheck a JSON catalogue design with the original native FEM.
//!
//! Reads `{"nodes": [[x, y], ...], "active": [[member, section], ...]}` from
//! standard input. The design must be one the tutorial's decoder can produce:
//! the reference ground structure's 18 nodes, supports and load nodes in place,
//! movable nodes within the decoder's offset limits, between `MIN_ACTIVE` and
//! `MAX_ACTIVE` distinct members and valid catalogue sections. It is then
//! analyzed with `fem::analyze` and evaluated under the scalar constraint
//! contract (`evaluate::evaluate_decoded`, training scenario, no removal
//! analysis), and the result is printed as JSON.
#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::io::{self, Read};
use truss_sizing::{
    catalogue::sections,
    decode::{ActiveMember, DecodedDesign, MAX_ACTIVE, MIN_ACTIVE},
    evaluate::evaluate_decoded,
    fem::{Scenario, WorkCounter, analyze},
    ground::{GroundStructure, Node},
};

type Error = Box<dyn std::error::Error>;

fn pair(value: &serde_json::Value, what: &str) -> Result<[serde_json::Value; 2], Error> {
    match value.as_array().map(Vec::as_slice) {
        Some([a, b]) => Ok([a.clone(), b.clone()]),
        _ => Err(format!("each {what} entry must have exactly two values").into()),
    }
}

fn parse(input: &str, ground: &GroundStructure) -> Result<DecodedDesign, Error> {
    let data: serde_json::Value = serde_json::from_str(input)?;
    let object = data.as_object().ok_or("a JSON object is required")?;
    if let Some(key) = object.keys().find(|k| *k != "nodes" && *k != "active") {
        return Err(format!("unknown key {key:?}").into());
    }
    let nodes = object
        .get("nodes")
        .and_then(serde_json::Value::as_array)
        .ok_or("nodes array required")?
        .iter()
        .map(|v| {
            let [x, y] = pair(v, "node")?;
            Ok(Node {
                x: x.as_f64().ok_or("invalid x")?,
                y: y.as_f64().ok_or("invalid y")?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if nodes.len() != ground.nodes.len() {
        return Err(format!("{} nodes required", ground.nodes.len()).into());
    }
    let dx_limit = 0.15 * ground.bay_m;
    let dy_limit = 0.15 * ground.level_m;
    for (index, (node, nominal)) in nodes.iter().zip(&ground.nodes).enumerate() {
        if !node.x.is_finite() || !node.y.is_finite() {
            return Err(format!("node {index} is not finite").into());
        }
        let (dx, dy) = ((node.x - nominal.x).abs(), (node.y - nominal.y).abs());
        let moved = dx > 1.0e-12 || dy > 1.0e-12;
        if ground.movable_nodes.contains(&index) {
            if dx > dx_limit + 1.0e-12 || dy > dy_limit + 1.0e-12 {
                return Err(format!("node {index} moves beyond the decoder's limits").into());
            }
        } else if moved {
            return Err(format!("node {index} is a support, load or fixed node").into());
        }
    }
    let catalogue = sections().len();
    let active = object
        .get("active")
        .and_then(serde_json::Value::as_array)
        .ok_or("active array required")?
        .iter()
        .map(|v| {
            let [member, section] = pair(v, "active")?;
            Ok(ActiveMember {
                member_index: usize::try_from(member.as_u64().ok_or("invalid member")?)?,
                section_index: usize::try_from(section.as_u64().ok_or("invalid section")?)?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if !(MIN_ACTIVE..=MAX_ACTIVE).contains(&active.len()) {
        return Err(format!("{MIN_ACTIVE} to {MAX_ACTIVE} active members required").into());
    }
    let distinct: BTreeSet<_> = active.iter().map(|m| m.member_index).collect();
    if distinct.len() != active.len() {
        return Err("active members must be distinct".into());
    }
    if active
        .iter()
        .any(|m| m.member_index >= ground.members.len() || m.section_index >= catalogue)
    {
        return Err("member or section index out of range".into());
    }
    Ok(DecodedDesign { nodes, active })
}

fn main() -> Result<(), Error> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let ground = GroundStructure::reference();
    let design = parse(&input, &ground)?;
    let counter = WorkCounter::default();
    let response = analyze(&ground, &design, Scenario::TRAINING, None, &counter);
    let value = match response {
        Ok(r) => {
            let evaluation = evaluate_decoded(
                Vec::new(),
                design,
                &ground,
                Scenario::TRAINING,
                false,
                &counter,
            );
            serde_json::json!({
                "status": "checked",
                "feasible": evaluation.feasible(),
                "mass_kg": evaluation.mass_kg,
                "stress": r.max_stress_ratio,
                "buckling": r.max_buckling_ratio,
                "displacement": r.max_displacement_m,
                "displacements": r.case_displacements,
                "forces": r.member_forces_n,
            })
        }
        Err(e) => serde_json::json!({"status": "rejected", "reason": format!("{e:?}")}),
    };
    println!("{}", serde_json::to_string(&value)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nominal_nodes(ground: &GroundStructure) -> Vec<[f64; 2]> {
        ground.nodes.iter().map(|n| [n.x, n.y]).collect()
    }

    fn design(nodes: &[[f64; 2]], active: &[[usize; 2]]) -> String {
        serde_json::json!({"nodes": nodes, "active": active}).to_string()
    }

    fn members(count: usize) -> Vec<[usize; 2]> {
        (0..count).map(|i| [i, 3]).collect()
    }

    #[test]
    fn accepts_a_decodable_design() {
        let ground = GroundStructure::reference();
        let nodes = nominal_nodes(&ground);
        assert!(parse(&design(&nodes, &members(MIN_ACTIVE)), &ground).is_ok());
    }

    #[test]
    fn rejects_designs_the_decoder_cannot_produce() {
        let ground = GroundStructure::reference();
        let nodes = nominal_nodes(&ground);
        let mut doubled = members(MIN_ACTIVE);
        doubled.push(doubled[0]);
        let mut moved_support = nodes.clone();
        moved_support[ground.pinned_node][0] += 0.1;
        let mut far = nodes.clone();
        far[ground.movable_nodes[0]][0] += 36.0;
        let mut bad_section = members(MIN_ACTIVE);
        bad_section[0][1] = sections().len();
        for (input, case) in [
            (design(&nodes, &doubled), "duplicate member"),
            (design(&nodes, &members(MIN_ACTIVE - 1)), "too few members"),
            (design(&nodes, &members(MAX_ACTIVE + 1)), "too many members"),
            (
                design(&moved_support, &members(MIN_ACTIVE)),
                "moved support",
            ),
            (
                design(&far, &members(MIN_ACTIVE)),
                "offset beyond the limit",
            ),
            (design(&nodes, &bad_section), "section out of range"),
            (
                serde_json::json!({"nodes": nodes, "active": members(MIN_ACTIVE), "x": 1})
                    .to_string(),
                "unknown key",
            ),
        ] {
            assert!(parse(&input, &ground).is_err(), "{case} was accepted");
        }
    }
}
