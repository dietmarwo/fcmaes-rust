//! Deterministic synthetic instances with replayable feasible witnesses.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Base customer count in publication instances.
pub const BASE_TASKS: usize = 50;
/// Reserve urgent tasks activated by one training scenario.
pub const RESERVE_TASKS: usize = 2;
/// Total fixed task superset.
pub const TASKS: usize = BASE_TASKS + RESERVE_TASKS;
/// Fleet size.
pub const VEHICLES: usize = 8;
/// Normalized decision dimension.
pub const DIMENSION: usize = 2 * TASKS;

/// One skill bit.
pub const SKILL_ELECTRICAL: u8 = 1;
/// One skill bit.
pub const SKILL_HVAC: u8 = 2;
/// One skill bit.
pub const SKILL_NETWORK: u8 = 4;

/// One service visit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Task {
    /// Stable task index.
    pub id: usize,
    /// East coordinate in kilometres.
    pub x_km: f64,
    /// North coordinate in kilometres.
    pub y_km: f64,
    /// On-site work in seconds.
    pub service_s: f64,
    /// Consumed vehicle capacity in kg-equivalent units.
    pub demand_kg: f64,
    /// Earliest service start.
    pub earliest_s: f64,
    /// Latest service start.
    pub latest_s: f64,
    /// Exactly one required skill bit.
    pub skill: u8,
    /// Whether this task belongs to the nominal base set.
    pub base: bool,
}

/// One technician vehicle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vehicle {
    /// Stable vehicle index.
    pub id: usize,
    /// Route capacity.
    pub capacity_kg: f64,
    /// Shift start.
    pub shift_start_s: f64,
    /// Shift end.
    pub shift_end_s: f64,
    /// Supported skill bit set.
    pub skills: u8,
    /// Fixed dispatch cost.
    pub fixed_cost: f64,
    /// Distance cost.
    pub cost_per_km: f64,
}

/// Frozen routing instance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Instance {
    /// Stable name.
    pub name: String,
    /// Generator seed.
    pub seed: u64,
    /// Depot east coordinate.
    pub depot_x_km: f64,
    /// Depot north coordinate.
    pub depot_y_km: f64,
    /// Constant speed.
    pub speed_km_h: f64,
    /// Complete task superset.
    pub tasks: Vec<Task>,
    /// Fleet.
    pub vehicles: Vec<Vehicle>,
    /// Known feasible route for base tasks, by vehicle.
    pub witness_routes: Vec<Vec<usize>>,
}

#[derive(Clone, Copy)]
struct Generator {
    state: u64,
}

impl Generator {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn range(&mut self, lower: f64, upper: f64) -> f64 {
        lower + (upper - lower) * self.unit()
    }
}

fn distance(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    (ax - bx).hypot(ay - by)
}

/// Generate a clustered instance from an explicit feasible witness.
#[must_use]
pub fn generate(seed: u64, index: usize) -> Instance {
    let mut rng = Generator::new(seed);
    let skill_sets = [
        SKILL_ELECTRICAL | SKILL_NETWORK,
        SKILL_HVAC | SKILL_ELECTRICAL,
        SKILL_NETWORK | SKILL_HVAC,
        SKILL_ELECTRICAL,
        SKILL_HVAC,
        SKILL_NETWORK,
        SKILL_ELECTRICAL | SKILL_HVAC,
        SKILL_ELECTRICAL | SKILL_HVAC | SKILL_NETWORK,
    ];
    let vehicles = (0..VEHICLES)
        .map(|id| Vehicle {
            id,
            capacity_kg: 380.0 + 30.0 * (id % 3) as f64,
            shift_start_s: 8.0 * 3600.0,
            shift_end_s: 20.0 * 3600.0,
            skills: skill_sets[id],
            fixed_cost: 72.0 + 5.0 * id as f64,
            cost_per_km: 0.78 + 0.035 * id as f64,
        })
        .collect::<Vec<_>>();
    let mut tasks = Vec::with_capacity(TASKS);
    let mut witness_routes = vec![Vec::new(); VEHICLES];
    let mut per_vehicle = [BASE_TASKS / VEHICLES; VEHICLES];
    for count in per_vehicle.iter_mut().take(BASE_TASKS % VEHICLES) {
        *count += 1;
    }
    for vehicle in 0..VEHICLES {
        let angle = 2.0 * std::f64::consts::PI * vehicle as f64 / VEHICLES as f64;
        let radius = 17.0 + 3.0 * (vehicle % 3) as f64;
        let center_x = radius * angle.cos();
        let center_y = radius * angle.sin();
        let mut previous = (0.0, 0.0);
        let mut clock = vehicles[vehicle].shift_start_s;
        let mut load = 0.0;
        for local in 0..per_vehicle[vehicle] {
            let local_angle = angle + 0.55 * (local as f64 - per_vehicle[vehicle] as f64 / 2.0);
            let spread = 2.0 + 3.5 * rng.unit();
            let x = center_x + spread * local_angle.cos() + rng.range(-1.2, 1.2);
            let y = center_y + spread * local_angle.sin() + rng.range(-1.2, 1.2);
            clock += 3600.0 * distance(previous.0, previous.1, x, y) / 48.0;
            let service = (rng.range(10.0, 27.0) * 60.0).round();
            let tight = local % 3 == 0;
            let earliest =
                (clock - if tight { 300.0 } else { 1200.0 }).max(vehicles[vehicle].shift_start_s);
            let latest = clock + if tight { 4.0 * 3600.0 } else { 7.0 * 3600.0 };
            let available = [SKILL_ELECTRICAL, SKILL_HVAC, SKILL_NETWORK]
                .into_iter()
                .filter(|skill| vehicles[vehicle].skills & skill != 0)
                .collect::<Vec<_>>();
            let skill = available[(rng.unit() * available.len() as f64) as usize % available.len()];
            let demand = rng.range(25.0, 46.0).round();
            load += demand;
            let id = tasks.len();
            tasks.push(Task {
                id,
                x_km: x,
                y_km: y,
                service_s: service,
                demand_kg: demand,
                earliest_s: earliest,
                latest_s: latest,
                skill,
                base: true,
            });
            witness_routes[vehicle].push(id);
            clock = clock.max(earliest) + service;
            previous = (x, y);
        }
        debug_assert!(load < vehicles[vehicle].capacity_kg);
        debug_assert!(
            clock + 3600.0 * distance(previous.0, previous.1, 0.0, 0.0) / 48.0
                < vehicles[vehicle].shift_end_s
        );
    }
    for reserve in 0..RESERVE_TASKS {
        let anchor = &tasks[witness_routes[reserve][2]];
        tasks.push(Task {
            id: BASE_TASKS + reserve,
            x_km: anchor.x_km + if reserve == 0 { 0.7 } else { -0.8 },
            y_km: anchor.y_km + if reserve == 0 { -0.5 } else { 0.6 },
            service_s: 12.0 * 60.0,
            demand_kg: 18.0,
            earliest_s: anchor.earliest_s + 300.0,
            latest_s: anchor.latest_s + 600.0,
            skill: anchor.skill,
            base: false,
        });
    }
    Instance {
        name: format!("fsr-{index:02}"),
        seed,
        depot_x_km: 0.0,
        depot_y_km: 0.0,
        speed_km_h: 48.0,
        tasks,
        vehicles,
        witness_routes,
    }
}

/// Ten frozen generator seeds.
pub const SEEDS: [u64; 10] = [11, 29, 47, 71, 101, 131, 173, 211, 257, 307];

/// Stable mixed-record CSV representation.
#[must_use]
pub fn to_csv(instance: &Instance) -> String {
    let mut out = String::from(
        "kind,id,x_or_capacity,y_or_shift_start,service_or_shift_end,demand_or_skills,earliest_or_fixed,latest_or_per_km,skill_or_task,base\n",
    );
    writeln!(
        out,
        "meta,{},17,0,0,{},{},{},{},{}",
        instance.name,
        instance.seed,
        instance.depot_x_km,
        instance.depot_y_km,
        instance.speed_km_h,
        index_from_name(&instance.name)
    )
    .expect("writing String cannot fail");
    for vehicle in &instance.vehicles {
        writeln!(
            out,
            "vehicle,{},{:.17},{:.17},{:.17},{},{:.17},{:.17},0,0",
            vehicle.id,
            vehicle.capacity_kg,
            vehicle.shift_start_s,
            vehicle.shift_end_s,
            vehicle.skills,
            vehicle.fixed_cost,
            vehicle.cost_per_km
        )
        .expect("writing String cannot fail");
    }
    for task in &instance.tasks {
        writeln!(
            out,
            "task,{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{},{}",
            task.id,
            task.x_km,
            task.y_km,
            task.service_s,
            task.demand_kg,
            task.earliest_s,
            task.latest_s,
            task.skill,
            usize::from(task.base)
        )
        .expect("writing String cannot fail");
    }
    for (vehicle, route) in instance.witness_routes.iter().enumerate() {
        for (order, task) in route.iter().enumerate() {
            writeln!(out, "witness,{vehicle},0,0,0,0,0,0,{task},{order}")
                .expect("writing String cannot fail");
        }
    }
    out
}

fn index_from_name(name: &str) -> usize {
    name.rsplit('-')
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// Parse the stable mixed-record CSV format.
pub fn from_csv(text: &str) -> Result<Instance, Box<dyn Error>> {
    let mut name = String::new();
    let mut seed = 0;
    let mut depot_x = 0.0;
    let mut depot_y = 0.0;
    let mut speed = 0.0;
    let mut tasks = Vec::new();
    let mut vehicles = Vec::new();
    let mut witness_routes = vec![Vec::new(); VEHICLES];
    for (line_index, line) in text.lines().enumerate().skip(1) {
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.len() != 10 {
            return Err(format!("line {} has {} fields", line_index + 1, fields.len()).into());
        }
        match fields[0] {
            "meta" => {
                name = fields[1].to_owned();
                seed = fields[5].parse()?;
                depot_x = fields[6].parse()?;
                depot_y = fields[7].parse()?;
                speed = fields[8].parse()?;
            }
            "vehicle" => vehicles.push(Vehicle {
                id: fields[1].parse()?,
                capacity_kg: fields[2].parse()?,
                shift_start_s: fields[3].parse()?,
                shift_end_s: fields[4].parse()?,
                skills: fields[5].parse()?,
                fixed_cost: fields[6].parse()?,
                cost_per_km: fields[7].parse()?,
            }),
            "task" => tasks.push(Task {
                id: fields[1].parse()?,
                x_km: fields[2].parse()?,
                y_km: fields[3].parse()?,
                service_s: fields[4].parse()?,
                demand_kg: fields[5].parse()?,
                earliest_s: fields[6].parse()?,
                latest_s: fields[7].parse()?,
                skill: fields[8].parse()?,
                base: fields[9] == "1",
            }),
            "witness" => {
                let vehicle: usize = fields[1].parse()?;
                let task: usize = fields[8].parse()?;
                let order: usize = fields[9].parse()?;
                if order != witness_routes[vehicle].len() {
                    return Err("witness records out of order".into());
                }
                witness_routes[vehicle].push(task);
            }
            kind => return Err(format!("unknown record kind {kind}").into()),
        }
    }
    Ok(Instance {
        name,
        seed,
        depot_x_km: depot_x,
        depot_y_km: depot_y,
        speed_km_h: speed,
        tasks,
        vehicles,
        witness_routes,
    })
}

/// Write all frozen instances.
pub fn write_instances(directory: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    for (index, seed) in SEEDS.iter().copied().enumerate() {
        fs::write(
            directory.join(format!("instance-{index:02}.csv")),
            to_csv(&generate(seed, index)),
        )?;
    }
    Ok(())
}

/// Load one checked-in instance, falling back to deterministic generation.
pub fn load_primary() -> Result<Instance, Box<dyn Error>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("instances/instance-00.csv");
    if path.exists() {
        from_csv(&fs::read_to_string(path)?)
    } else {
        Ok(generate(SEEDS[0], 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_round_trip_is_exact_and_deterministic() {
        let instance = generate(11, 0);
        let encoded = to_csv(&instance);
        assert_eq!(encoded, to_csv(&generate(11, 0)));
        assert_eq!(instance, from_csv(&encoded).unwrap());
    }

    #[test]
    fn generator_has_fixed_superset_and_mixed_windows() {
        for (index, seed) in SEEDS.iter().copied().enumerate() {
            let instance = generate(seed, index);
            assert_eq!(instance.tasks.len(), TASKS);
            assert_eq!(
                instance.tasks.iter().filter(|task| task.base).count(),
                BASE_TASKS
            );
            assert_eq!(instance.vehicles.len(), VEHICLES);
            let widths = instance.tasks[..BASE_TASKS]
                .iter()
                .map(|task| task.latest_s - task.earliest_s)
                .collect::<Vec<_>>();
            assert!(widths.iter().any(|width| *width < 15_000.0));
            assert!(widths.iter().any(|width| *width > 25_000.0));
            assert!(instance.tasks.iter().all(|task| {
                instance
                    .vehicles
                    .iter()
                    .any(|vehicle| vehicle.skills & task.skill != 0)
            }));
        }
    }
}
