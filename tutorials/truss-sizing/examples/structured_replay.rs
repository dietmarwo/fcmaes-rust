//! Recheck a JSON catalogue design with the original native FEM.
use std::io::{self, Read};
use truss_sizing::{
    decode::{ActiveMember, DecodedDesign},
    fem::{Scenario, WorkCounter, analyze},
    ground::{GroundStructure, Node},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let data: serde_json::Value = serde_json::from_str(&input)?;
    let ground = GroundStructure::reference();
    let nodes = data["nodes"]
        .as_array()
        .ok_or("nodes array required")?
        .iter()
        .map(|v| {
            Ok(Node {
                x: v[0].as_f64().ok_or("invalid x")?,
                y: v[1].as_f64().ok_or("invalid y")?,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let active = data["active"]
        .as_array()
        .ok_or("active array required")?
        .iter()
        .map(|v| {
            Ok(ActiveMember {
                member_index: v[0].as_u64().ok_or("invalid member")? as usize,
                section_index: v[1].as_u64().ok_or("invalid section")? as usize,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    if nodes.len() != ground.nodes.len()
        || nodes.iter().any(|n| !n.x.is_finite() || !n.y.is_finite())
        || active
            .iter()
            .any(|m| m.member_index >= ground.members.len() || m.section_index >= 12)
    {
        return Err("invalid design dimensions or indices".into());
    }
    let response = analyze(
        &ground,
        &DecodedDesign { nodes, active },
        Scenario::TRAINING,
        None,
        &WorkCounter::default(),
    );
    let value = match response {
        Ok(r) => serde_json::json!({"status":"checked","stress":r.max_stress_ratio,
            "buckling":r.max_buckling_ratio,"displacement":r.max_displacement_m,
            "displacements":r.case_displacements,"forces":r.member_forces_n}),
        Err(e) => serde_json::json!({"status":"rejected","reason":format!("{e:?}")}),
    };
    println!("{}", serde_json::to_string(&value)?);
    Ok(())
}
