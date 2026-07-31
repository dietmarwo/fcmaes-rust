//! Conditional quality-diversity campaign.

use crate::pilot::QdDecision;

/// QD execution outcome after the registered pilot gate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QdOutcome {
    /// The archive was deliberately not executed.
    Skipped {
        /// Machine-readable reason.
        reason: &'static str,
    },
}

/// Apply the frozen M8 execution gate.
#[must_use]
pub const fn gate(decision: QdDecision) -> QdOutcome {
    match decision {
        QdDecision::Accepted => QdOutcome::Skipped {
            reason: "primary descriptor accepted, but no publication QD budget was authorized",
        },
        QdDecision::PrimarySecondary => QdOutcome::Skipped {
            reason: "primary descriptor failed; fallback is supporting evidence only",
        },
        QdDecision::Rejected => QdOutcome::Skipped {
            reason: "descriptor pilot rejected both emergent pairs",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_pilot_has_explicit_skip_reason() {
        assert!(matches!(
            gate(QdDecision::Rejected),
            QdOutcome::Skipped {
                reason: "descriptor pilot rejected both emergent pairs"
            }
        ));
    }
}
