//! Deduplicated result archive and motif-rediscovery accounting.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::grammar::REFERENCES;
use crate::inner::CandidateResult;

/// All accepted, numerically evaluated proposals for one arm.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Archive {
    pub candidates: Vec<CandidateResult>,
}

impl Archive {
    /// Add a new topology; duplicates consume no inner budget.
    pub fn insert(&mut self, candidate: CandidateResult) -> bool {
        if self.contains_key(&candidate.topology_key) {
            false
        } else {
            self.candidates.push(candidate);
            true
        }
    }

    /// Has the canonical topology already been evaluated?
    pub fn contains_key(&self, key: &str) -> bool {
        self.candidates
            .iter()
            .any(|candidate| candidate.topology_key == key)
    }

    /// Best holdout candidate.
    pub fn best(&self) -> Option<&CandidateResult> {
        self.candidates.iter().min_by(|left, right| {
            left.validation
                .scalar_score
                .total_cmp(&right.validation.scalar_score)
        })
    }

    /// First exact rediscovery proposal for each held-out reference.
    pub fn exact_rediscoveries(&self) -> BTreeMap<String, usize> {
        REFERENCES
            .iter()
            .map(|(name, reference)| {
                let found = self
                    .candidates
                    .iter()
                    .find(|candidate| candidate.topology == *reference)
                    .map(|candidate| candidate.proposal);
                ((*name).to_owned(), found.unwrap_or(0))
            })
            .collect()
    }

    /// Motif-class coverage independent of exact reference orientation.
    pub fn motif_classes(&self) -> BTreeSet<String> {
        self.candidates
            .iter()
            .flat_map(|candidate| candidate.motif_flags.clone())
            .filter(|label| label != "other")
            .collect()
    }

    /// Write a crash-replay JSONL archive.
    pub fn write_jsonl(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let mut output = String::new();
        for candidate in &self.candidates {
            output.push_str(&serde_json::to_string(candidate)?);
            output.push('\n');
        }
        fs::write(path, output)?;
        Ok(())
    }

    /// Restore an archive written by [`write_jsonl`](Self::write_jsonl).
    pub fn read_jsonl(path: &Path) -> Result<Self, Box<dyn Error>> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let candidates = fs::read_to_string(path)?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { candidates })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Preset, Protocol};
    use crate::inner::optimize;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn jsonl_round_trip_and_deduplication() {
        let protocol = Protocol {
            inner_retries: 1,
            workers: 1,
            inner_evaluations: 8,
            ..Protocol::for_preset(Preset::Smoke)
        };
        let candidate = optimize(REFERENCES[0].1, "test", 1, &protocol, 9);
        let mut archive = Archive::default();
        assert!(archive.insert(candidate.clone()));
        assert!(!archive.insert(candidate));
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("oscillator-archive-{suffix}.jsonl"));
        archive.write_jsonl(&path).unwrap();
        let replay = Archive::read_jsonl(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(replay.candidates.len(), 1);
        assert_eq!(replay.candidates[0].topology_key, REFERENCES[0].1.key());
    }
}
