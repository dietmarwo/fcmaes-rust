//! Print history-blind bounds and a valid neutral schedule for an agent proposal.
use gtoc1_pykep::job_models::route_and_bounds;
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let text = std::env::args()
        .nth(1)
        .ok_or("usage: gtoc1-route-card 3,2,3,5,10")?;
    if text == "--help" || text == "-h" {
        println!(
            "usage: gtoc1-route-card 3,2,3,5,10\nEmits JSON bounds and a neutral schedule for a grammar-valid body order.\nEarth=3, Venus=2, Jupiter=5, Saturn=6, asteroid=10. Impulsive MGA screening only."
        );
        return Ok(());
    }
    let bodies = text
        .split(',')
        .map(str::parse)
        .collect::<Result<Vec<usize>, _>>()?;
    let (route, lo, hi) = route_and_bounds(bodies.clone())?;
    let b = route.codec().optimizer_bounds();
    let midpoint: Vec<f64> = b
        .lower()
        .iter()
        .zip(b.upper())
        .map(|(a, b)| a.midpoint(*b))
        .collect();
    let guess = route.codec().decode(&midpoint)?.as_sequence_decision();
    println!(
        "{}",
        serde_json::json!({"bodies":bodies,"lower":lo,"upper":hi,"guess":guess,
        "bounds_profile":"route-derived-direct","claim_boundary":"impulsive MGA screening only"})
    );
    Ok(())
}
