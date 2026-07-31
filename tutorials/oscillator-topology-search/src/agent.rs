//! Provider-independent subprocess boundary for topology proposals.

use std::error::Error;
use std::fmt;
use std::io::Write;
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use crate::archive::Archive;
use crate::grammar::Topology;

/// Compact archive information sent to a proposer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentObservation {
    pub proposal_attempt: usize,
    pub grammar: &'static str,
    pub objective: &'static str,
    pub evaluated: Vec<ObservedCandidate>,
    pub rejected_keys: Vec<String>,
    pub repair_error: Option<String>,
}

/// Score summary with no kinetic vector or held-out reference list.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservedCandidate {
    pub topology: String,
    pub validation_score: f64,
    pub motif_flags: Vec<String>,
    pub dimension: usize,
}

#[derive(Debug, Deserialize)]
struct AgentProposal {
    edges: Vec<u8>,
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

/// Validated proposal and provider-reported usage.
#[derive(Clone, Debug)]
pub struct AgentOutcome {
    pub topology: Topology,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Failure class used by the campaign circuit breaker and accounting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentErrorKind {
    /// The subprocess could not be started or exited unsuccessfully.
    Transport,
    /// The subprocess returned malformed or grammar-invalid output twice.
    InvalidResponse,
}

/// Typed proposal-adapter failure with a bounded diagnostic.
#[derive(Debug)]
pub struct AgentError {
    kind: AgentErrorKind,
    detail: String,
}

impl AgentError {
    fn new(kind: AgentErrorKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into().chars().take(1_000).collect(),
        }
    }

    /// Transport versus invalid-response classification.
    pub fn kind(&self) -> AgentErrorKind {
        self.kind
    }
}

impl fmt::Display for AgentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.detail)
    }
}

impl Error for AgentError {}

fn invoke(command: &str, observation: &AgentObservation) -> Result<AgentOutcome, AgentError> {
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| AgentError::new(AgentErrorKind::Transport, error.to_string()))?;
    let request = serde_json::to_vec(observation)
        .map_err(|error| AgentError::new(AgentErrorKind::InvalidResponse, error.to_string()))?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| AgentError::new(AgentErrorKind::Transport, "agent stdin unavailable"))?
        .write_all(&request)
        .map_err(|error| AgentError::new(AgentErrorKind::Transport, error.to_string()))?;
    let output = child
        .wait_with_output()
        .map_err(|error| AgentError::new(AgentErrorKind::Transport, error.to_string()))?;
    if !output.status.success() {
        return Err(AgentError::new(
            AgentErrorKind::Transport,
            format!(
                "agent command failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ));
    }
    let proposal: AgentProposal = serde_json::from_slice(&output.stdout).map_err(|error| {
        AgentError::new(
            AgentErrorKind::InvalidResponse,
            format!("invalid agent JSON: {error}"),
        )
    })?;
    let edges: [u8; 9] = proposal.edges.try_into().map_err(|_| {
        AgentError::new(
            AgentErrorKind::InvalidResponse,
            "agent proposal needs exactly nine edges",
        )
    })?;
    Ok(AgentOutcome {
        topology: Topology::try_new(edges)
            .map_err(|error| AgentError::new(AgentErrorKind::InvalidResponse, error))?,
        input_tokens: proposal.input_tokens,
        output_tokens: proposal.output_tokens,
    })
}

/// Invoke once and permit one bounded repair request for invalid output.
pub fn propose(
    command: &str,
    attempt: usize,
    archive: &Archive,
    rejected_keys: &[String],
) -> Result<AgentOutcome, AgentError> {
    let evaluated = archive
        .candidates
        .iter()
        .map(|candidate| ObservedCandidate {
            topology: candidate.topology_key.clone(),
            validation_score: candidate.validation.scalar_score,
            motif_flags: candidate.motif_flags.clone(),
            dimension: candidate.parameter_dimension,
        })
        .collect();
    let mut observation = AgentObservation {
        proposal_attempt: attempt,
        grammar: "nine digits in {0,1,2}; 2..=6 active; no isolated gene",
        objective: "minimize holdout oscillator score; seek distinct signed topologies",
        evaluated,
        rejected_keys: rejected_keys.to_vec(),
        repair_error: None,
    };
    match invoke(command, &observation) {
        Ok(topology) => Ok(topology),
        Err(error) if error.kind() == AgentErrorKind::InvalidResponse => {
            observation.repair_error = Some(error.to_string());
            invoke(command, &observation)
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::exact_reference;

    #[test]
    fn checked_in_mock_contains_no_reference() {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("agents")
            .join("mock_agent.py");
        let command = format!("python3 '{}'", script.display());
        let outcome = propose(&command, 1, &Archive::default(), &[]).unwrap();
        assert!(exact_reference(&outcome.topology).is_none());
    }

    #[test]
    fn malformed_output_gets_one_bounded_repair() {
        let command = r#"python3 -c 'import json,sys; r=json.load(sys.stdin); print("{\"edges\":[1,1,1,0,0,0,0,0,0]}" if r.get("repair_error") else "not-json")'"#;
        let outcome = propose(command, 1, &Archive::default(), &[]).unwrap();
        assert_eq!(outcome.topology.key(), "111000000");

        let always_bad = r#"python3 -c 'import sys; sys.stdin.read(); print("not-json")'"#;
        let error = propose(always_bad, 1, &Archive::default(), &[]).unwrap_err();
        assert_eq!(error.kind(), AgentErrorKind::InvalidResponse);
    }

    #[test]
    fn failed_command_is_not_retried_as_a_format_repair() {
        let error = propose("false", 1, &Archive::default(), &[]).unwrap_err();
        assert_eq!(error.kind(), AgentErrorKind::Transport);
    }
}
