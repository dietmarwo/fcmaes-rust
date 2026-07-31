//! Stable publication artifact schema and writers.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde_json::json;

use crate::config::Protocol;
use crate::coverage::{GROUP_WEIGHT_EXPONENT, evaluate, marginal_greedy};
use crate::instance::{FIXTURES, Instance, generate};
use crate::mo::MoResult;
use crate::oracle::{exact_cover, matching_cover, verify_cover, weighted_primal_dual};
use crate::so::{SoObjective, SoResult};
use crate::throughput::ThroughputResult;

/// Artifact schema version.
pub const SCHEMA_VERSION: u32 = 1;

/// Metadata common to all run manifests.
pub struct RunMetadata<'a> {
    /// Artifact directory.
    pub directory: &'a Path,
    /// Exact replay command.
    pub command: &'a str,
    /// Root seed.
    pub seed: u64,
    /// Requested worker count.
    pub workers: i32,
}

fn write(path: &Path, contents: &str) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, contents)?;
    Ok(())
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<(), Box<dyn Error>> {
    write(path, &(serde_json::to_string_pretty(value)? + "\n"))
}

fn mask(values: &[bool]) -> String {
    values
        .iter()
        .enumerate()
        .filter(|(_, chosen)| **chosen)
        .map(|(index, _)| index.to_string())
        .collect::<Vec<_>>()
        .join(";")
}

/// Write fixture summaries, independent certificates, exact tiny results, and
/// group-exponent sensitivity.
pub fn write_validation(metadata: &RunMetadata<'_>) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut instances =
        String::from("instance,nodes,edges,groups,group_memberships,connected,total_cost\n");
    let mut certificates = String::from(
        "instance,objective,method,selected,cover_value,lower_bound,ratio,verified,exact_status,exact_optimum\n",
    );
    for (index, config) in FIXTURES.iter().enumerate() {
        let instance = generate(config)?;
        writeln!(
            instances,
            "{},{},{},{},{},{},{:.17}",
            config.name,
            instance.nodes(),
            instance.edges.len(),
            instance.groups.len(),
            instance.groups.iter().map(Vec::len).sum::<usize>(),
            usize::from(instance.connected()),
            instance.costs.iter().sum::<f64>()
        )?;
        let matching = matching_cover(&instance);
        let cardinality = matching.selected.iter().filter(|value| **value).count();
        let exact_cardinality = if index == 0 {
            Some(exact_cover(&instance, false)?.objective)
        } else {
            None
        };
        writeln!(
            certificates,
            "{},cardinality,maximal-matching-endpoints,{},{},{},{:.17},1,{},{}",
            config.name,
            cardinality,
            cardinality,
            matching.lower_bound,
            cardinality as f64 / matching.lower_bound.max(1) as f64,
            if exact_cardinality.is_some() {
                "optimal"
            } else {
                "not-attempted"
            },
            exact_cardinality.map_or_else(String::new, |value| format!("{value:.17}"))
        )?;
        let weighted = weighted_primal_dual(&instance);
        let exact_weighted = if index == 0 {
            Some(exact_cover(&instance, true)?.objective)
        } else {
            None
        };
        writeln!(
            certificates,
            "{},weighted,primal-dual,{},{:.17},{:.17},{:.17},1,{},{}",
            config.name,
            weighted.selected.iter().filter(|value| **value).count(),
            weighted.cost,
            weighted.lower_bound,
            weighted.cost / weighted.lower_bound.max(1.0e-30),
            if exact_weighted.is_some() {
                "optimal"
            } else {
                "not-attempted"
            },
            exact_weighted.map_or_else(String::new, |value| format!("{value:.17}"))
        )?;
    }
    let tiny = generate(&FIXTURES[0])?;
    let greedy = marginal_greedy(&tiny, GROUP_WEIGHT_EXPONENT);
    let mut sensitivity = String::from("exponent,prefix,selected,cost,coverage,roi\n");
    for exponent in [0.0, 0.5, 1.0] {
        for index in [tiny.nodes() / 4, tiny.nodes() / 2, 3 * tiny.nodes() / 4] {
            let selected = &greedy[index].metrics.selected;
            let metrics = evaluate(&tiny, selected, exponent).ok_or("sensitivity replay failed")?;
            writeln!(
                sensitivity,
                "{exponent:.1},{index},{},{:.17},{:.17},{:.17}",
                metrics.selected_count, metrics.cost, metrics.coverage, metrics.roi
            )?;
        }
    }
    write(&metadata.directory.join("instances.csv"), &instances)?;
    write(
        &metadata.directory.join("classic_oracles.csv"),
        &certificates,
    )?;
    write(
        &metadata.directory.join("group_weight_sensitivity.csv"),
        &sensitivity,
    )?;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": SCHEMA_VERSION,
            "tutorial": "network-coverage",
            "formulation": "validation",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": 0,
            "actual_evaluations": 0,
            "group_weight_exponent": GROUP_WEIGHT_EXPONENT,
            "exact_scope": "tiny only; larger rows carry certified lower bounds, not exact labels",
            "artifacts": {
                "instances": "instances.csv",
                "classic_oracles": "classic_oracles.csv",
                "group_weight_sensitivity": "group_weight_sensitivity.csv"
            }
        }),
    )
}

/// Write the pre-optimization throughput reconnaissance.
pub fn write_throughput(
    metadata: &RunMetadata<'_>,
    rows: &[ThroughputResult],
    selected_instance: &str,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut csv = String::from(
        "instance,samples,workers,elapsed_seconds,candidates_per_second,edge_visits_per_second,group_memberships_per_second,checksum\n",
    );
    for row in rows {
        writeln!(
            csv,
            "{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17}",
            row.instance,
            row.samples,
            row.workers,
            row.elapsed.as_secs_f64(),
            row.candidates_per_second,
            row.edge_visits_per_second,
            row.group_memberships_per_second,
            row.checksum
        )?;
    }
    write(&metadata.directory.join("throughput.csv"), &csv)?;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": SCHEMA_VERSION,
            "tutorial": "network-coverage",
            "formulation": "throughput",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": rows.iter().map(|row| row.samples).sum::<usize>(),
            "actual_evaluations": rows.iter().map(|row| row.samples).sum::<usize>(),
            "selected_publication_instance": selected_instance,
            "gate": "reference-4k only when its serial kernel reaches 20,000 candidates/s",
            "interpretation": "implementation diagnostic; not a cross-library benchmark",
            "artifacts": {"throughput": "throughput.csv"}
        }),
    )
}

/// Write certified baselines and both scalar DE arms.
pub fn write_so(
    metadata: &RunMetadata<'_>,
    instance: &Instance,
    results: &[SoResult],
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let matching = matching_cover(instance);
    let weighted = weighted_primal_dual(instance);
    let mut summary = String::from(
        "arm,objective,selected_count,cost,uncovered_edges,verified,retained_source,delta_vs_seed,lower_bound,ratio_to_bound,requested_evaluations,actual_evaluations,elapsed_seconds,coverage,roi,selected_nodes\n",
    );
    let matching_metrics = evaluate(instance, &matching.selected, GROUP_WEIGHT_EXPONENT)
        .ok_or("matching replay failed")?;
    writeln!(
        summary,
        "matching-endpoints,cardinality,{},{:.17},0,1,construction,0,{},{:.17},0,0,0,{:.17},{:.17},\"{}\"",
        matching_metrics.selected_count,
        matching_metrics.cost,
        matching.lower_bound,
        matching_metrics.selected_count as f64 / matching.lower_bound.max(1) as f64,
        matching_metrics.coverage,
        matching_metrics.roi,
        mask(&matching_metrics.selected)
    )?;
    let weighted_metrics = evaluate(instance, &weighted.selected, GROUP_WEIGHT_EXPONENT)
        .ok_or("primal-dual replay failed")?;
    writeln!(
        summary,
        "primal-dual,weighted,{},{:.17},0,1,construction,0,{:.17},{:.17},0,0,0,{:.17},{:.17},\"{}\"",
        weighted_metrics.selected_count,
        weighted_metrics.cost,
        weighted.lower_bound,
        weighted_metrics.cost / weighted.lower_bound.max(1.0e-30),
        weighted_metrics.coverage,
        weighted_metrics.roi,
        mask(&weighted_metrics.selected)
    )?;
    let mut convergence = String::from("arm,evaluations,elapsed_seconds,best_objective\n");
    let mut optimizer_incumbents = String::from(
        "arm,objective,metrics_available,objective_value,selected_count,cost,uncovered_edges,verified,coverage,roi,selected_nodes\n",
    );
    for result in results {
        let (bound, cover_value) = match result.objective {
            SoObjective::Cardinality => (
                matching.lower_bound as f64,
                result.best.selected_count as f64,
            ),
            SoObjective::Weighted => (weighted.lower_bound, result.best.cost),
        };
        writeln!(
            summary,
            "{},{},{},{:.17},{},{},{},{:.17},{:.17},{:.17},{},{},{:.17},{:.17},{:.17},\"{}\"",
            result.objective.name(),
            match result.objective {
                SoObjective::Cardinality => "cardinality",
                SoObjective::Weighted => "weighted",
            },
            result.best.selected_count,
            result.best.cost,
            result.best.uncovered_edges,
            usize::from(result.verified && verify_cover(instance, &result.best.selected)),
            result.retained_source,
            result.delta_vs_seed,
            bound,
            cover_value / bound.max(1.0e-30),
            result.requested_evaluations,
            result.actual_evaluations,
            result.elapsed.as_secs_f64(),
            result.best.coverage,
            result.best.roi,
            mask(&result.best.selected)
        )?;
        if let Some(incumbent) = &result.optimizer_incumbent {
            writeln!(
                optimizer_incumbents,
                "{},{},1,{:.17},{},{:.17},{},{},{:.17},{:.17},\"{}\"",
                result.objective.name(),
                match result.objective {
                    SoObjective::Cardinality => "cardinality",
                    SoObjective::Weighted => "weighted",
                },
                result.optimizer_objective,
                incumbent.selected_count,
                incumbent.cost,
                incumbent.uncovered_edges,
                usize::from(verify_cover(instance, &incumbent.selected)),
                incumbent.coverage,
                incumbent.roi,
                mask(&incumbent.selected)
            )?;
        } else {
            writeln!(
                optimizer_incumbents,
                "{},{},0,{:.17},,,,,,,\"\"",
                result.objective.name(),
                match result.objective {
                    SoObjective::Cardinality => "cardinality",
                    SoObjective::Weighted => "weighted",
                },
                result.optimizer_objective
            )?;
        }
        if result.improvements.is_empty() {
            writeln!(
                convergence,
                "{},{},{:.17},{:.17}",
                result.objective.name(),
                result.actual_evaluations,
                result.elapsed.as_secs_f64(),
                cover_value
            )?;
        } else {
            for row in &result.improvements {
                writeln!(
                    convergence,
                    "{},{},{:.17},{:.17}",
                    result.objective.name(),
                    row.evaluations,
                    row.elapsed_seconds,
                    row.value
                )?;
            }
        }
    }
    write(&metadata.directory.join("arms.csv"), &summary)?;
    write(
        &metadata.directory.join("optimizer_incumbents.csv"),
        &optimizer_incumbents,
    )?;
    write(&metadata.directory.join("convergence.csv"), &convergence)?;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": SCHEMA_VERSION,
            "tutorial": "network-coverage",
            "formulation": "so",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "instance": instance.metadata.name,
            "input": {
                "external_edges": instance.metadata.external_edges,
                "dropped_self_loops": instance.metadata.dropped_self_loops
            },
            "requested_evaluations": results.iter().map(|row| row.requested_evaluations).sum::<u64>(),
            "actual_evaluations": results.iter().map(|row| row.actual_evaluations).sum::<u64>(),
            "certificate_contract": "cardinality and weighted ratios use distinct valid lower bounds",
            "retention": results.iter().map(|result| json!({
                "arm": result.objective.name(),
                "retained_source": result.retained_source,
                "delta_vs_seed": result.delta_vs_seed,
                "optimizer_objective": result.optimizer_objective,
                "optimizer_metrics_available": result.optimizer_incumbent.is_some(),
                "optimizer_verified": result.optimizer_incumbent.as_ref().is_some_and(|metrics| verify_cover(instance, &metrics.selected))
            })).collect::<Vec<_>>(),
            "artifacts": {
                "arms": "arms.csv",
                "optimizer_incumbents": "optimizer_incumbents.csv",
                "convergence": "convergence.csv"
            }
        }),
    )
}

/// Write MODE, greedy, and convergence artifacts.
pub fn write_mo(
    metadata: &RunMetadata<'_>,
    instance: &Instance,
    result: &MoResult,
    sensitivity: Option<&MoResult>,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut pareto = String::from(
        "source,selected_for_documentation,selected_count,cost,edge_coverage,group_coverage,coverage,roi,uncovered_edges,selected_nodes\n",
    );
    for point in &result.pareto {
        writeln!(
            pareto,
            "{},{},{},{:.17},{},{:.17},{:.17},{:.17},{},\"{}\"",
            point.origin,
            usize::from(point.selected),
            point.metrics.selected_count,
            point.metrics.cost,
            point.metrics.edge_coverage,
            point.metrics.group_coverage,
            point.metrics.coverage,
            point.metrics.roi,
            point.metrics.uncovered_edges,
            mask(&point.metrics.selected)
        )?;
    }
    let mut greedy = String::from(
        "source,step,selected_count,cost,edge_coverage,group_coverage,coverage,roi,uncovered_edges\n",
    );
    for (step, point) in result.greedy.iter().enumerate() {
        writeln!(
            greedy,
            "marginal-greedy,{step},{},{:.17},{},{:.17},{:.17},{:.17},{}",
            point.selected_count,
            point.cost,
            point.edge_coverage,
            point.group_coverage,
            point.coverage,
            point.roi,
            point.uncovered_edges
        )?;
    }
    let mut prefixes = String::from(
        "source,step,selected_count,cost,edge_coverage,group_coverage,coverage,roi,uncovered_edges\n",
    );
    for (step, point) in result.greedy_prefixes.iter().enumerate() {
        writeln!(
            prefixes,
            "marginal-greedy-prefix,{step},{},{:.17},{},{:.17},{:.17},{:.17},{}",
            point.selected_count,
            point.cost,
            point.edge_coverage,
            point.group_coverage,
            point.coverage,
            point.roi,
            point.uncovered_edges
        )?;
    }
    let mut convergence = String::from("evaluations,elapsed_seconds,pareto_population\n");
    for row in &result.progress {
        writeln!(
            convergence,
            "{},{:.17},{}",
            row.evaluations, row.elapsed_seconds, row.pareto_population
        )?;
    }
    let not_dominated_by_greedy = |campaign: &MoResult| {
        campaign
            .pareto
            .iter()
            .filter(|point| {
                !campaign.greedy_prefixes.iter().any(|greedy| {
                    greedy.cost <= point.metrics.cost
                        && greedy.roi >= point.metrics.roi
                        && (greedy.cost < point.metrics.cost || greedy.roi > point.metrics.roi)
                })
            })
            .count()
    };
    let generated_not_dominated_by_greedy = |campaign: &MoResult| {
        campaign
            .pareto
            .iter()
            .filter(|point| point.origin == "mode-generated")
            .filter(|point| {
                !campaign.greedy_prefixes.iter().any(|greedy| {
                    greedy.cost <= point.metrics.cost
                        && greedy.roi >= point.metrics.roi
                        && (greedy.cost < point.metrics.cost || greedy.roi > point.metrics.roi)
                })
            })
            .count()
    };
    let campaigns = std::iter::once(("publication", result))
        .chain(sensitivity.map(|campaign| ("high-budget", campaign)));
    let mut budget_sensitivity = String::from(
        "campaign,requested_evaluations,actual_evaluations,elapsed_seconds,mode_front_size,mode_generated_points,mode_retained_initial_points,mode_points_not_dominated_by_greedy,mode_generated_not_dominated_by_greedy\n",
    );
    let mut budget_json = Vec::new();
    for (label, campaign) in campaigns {
        let generated = campaign
            .pareto
            .iter()
            .filter(|point| point.origin == "mode-generated")
            .count();
        let retained = campaign.pareto.len() - generated;
        let survivors = not_dominated_by_greedy(campaign);
        let generated_survivors = generated_not_dominated_by_greedy(campaign);
        writeln!(
            budget_sensitivity,
            "{label},{},{},{:.17},{},{},{},{},{}",
            campaign.requested_evaluations,
            campaign.actual_evaluations,
            campaign.elapsed.as_secs_f64(),
            campaign.pareto.len(),
            generated,
            retained,
            survivors,
            generated_survivors
        )?;
        budget_json.push(json!({
            "campaign": label,
            "requested_evaluations": campaign.requested_evaluations,
            "actual_evaluations": campaign.actual_evaluations,
            "elapsed_seconds": campaign.elapsed.as_secs_f64(),
            "mode_front_size": campaign.pareto.len(),
            "mode_generated_points": generated,
            "mode_retained_initial_points": retained,
            "mode_points_not_dominated_by_greedy": survivors,
            "mode_generated_not_dominated_by_greedy": generated_survivors
        }));
    }
    write(&metadata.directory.join("pareto.csv"), &pareto)?;
    write(&metadata.directory.join("greedy_front.csv"), &greedy)?;
    write(&metadata.directory.join("greedy_prefixes.csv"), &prefixes)?;
    write(&metadata.directory.join("convergence.csv"), &convergence)?;
    write(
        &metadata.directory.join("budget_sensitivity.csv"),
        &budget_sensitivity,
    )?;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": SCHEMA_VERSION,
            "tutorial": "network-coverage",
            "formulation": "mo",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "instance": instance.metadata.name,
            "input": {
                "external_edges": instance.metadata.external_edges,
                "dropped_self_loops": instance.metadata.dropped_self_loops
            },
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": ["normalized_cost", "one_minus_roi"],
            "mode_front_size": result.pareto.len(),
            "greedy_front_size": result.greedy.len(),
            "greedy_prefix_count": result.greedy_prefixes.len(),
            "budget_sensitivity": budget_json,
            "artifacts": {
                "pareto": "pareto.csv",
                "greedy_front": "greedy_front.csv",
                "greedy_prefixes": "greedy_prefixes.csv",
                "convergence": "convergence.csv",
                "budget_sensitivity": "budget_sensitivity.csv"
            }
        }),
    )
}

/// Write the frozen effective protocol after the throughput gate.
pub fn write_protocol(
    path: &Path,
    preset: &str,
    selected_instance: &str,
    protocol: Protocol,
    evaluations_override: Option<usize>,
    sensitivity_enabled: bool,
) -> Result<(), Box<dyn Error>> {
    let so_evaluations = evaluations_override.map_or(protocol.so_evaluations, |value| value as u64);
    let mo_evaluations = evaluations_override.unwrap_or(protocol.mo_evaluations);
    let mo_sensitivity_evaluations = if sensitivity_enabled {
        protocol.mo_sensitivity_evaluations
    } else {
        0
    };
    write_json(
        path,
        &json!({
            "schema_version": SCHEMA_VERSION,
            "tutorial": "network-coverage",
            "preset": preset,
            "selected_instance": selected_instance,
            "selection_precedes_optimization": true,
            "so_evaluations_per_arm": so_evaluations,
            "so_retries": protocol.so_retries,
            "mo_evaluations": mo_evaluations,
            "mo_sensitivity_evaluations": mo_sensitivity_evaluations,
            "mo_population": protocol.mo_population,
            "group_weight_exponent": GROUP_WEIGHT_EXPONENT,
            "decision_bounds": [0.0, 1.999999999999],
            "integer_mask": true
        }),
    )
}
