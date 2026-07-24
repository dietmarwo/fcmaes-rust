//! Component-oriented NeXosim production-line model.

use std::collections::VecDeque;
use std::time::Duration;

use nexosim::Message;
use nexosim::model::{Context, Model, schedulable};
use nexosim::ports::{EventSinkReader, EventSource, Output, SinkState, event_queue};
use nexosim::simulation::{Mailbox, SimInit};
use nexosim::time::MonotonicTime;
use serde::{Deserialize, Serialize};

/// MODE decision-space dimension.
pub const DIM: usize = 9;
/// Number of minimized objectives.
pub const OBJECTIVES: usize = 4;
/// Lower decision bounds.
pub const LOWER: [f64; DIM] = [1.0, 0.70, 0.70, 0.15, 0.15, 0.0, 0.0, 1.0, 1.0];
/// Upper decision bounds.
pub const UPPER: [f64; DIM] = [32.0, 1.60, 1.60, 0.95, 0.95, 1.0, 1.0, 4.0, 4.0];
/// Integer-coordinate mask used by MODE.
pub const INTEGERS: [bool; DIM] = [true, false, false, false, false, false, false, true, true];

const MAX_REWORKS: u8 = 2;

/// Decoded production-line controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Design {
    pub buffer_capacity: usize,
    pub speed_a: f64,
    pub speed_b: f64,
    pub maintenance_threshold_a: f64,
    pub maintenance_threshold_b: f64,
    pub rework_probability: f64,
    pub dispatch_priority: f64,
    pub staff_a: usize,
    pub staff_b: usize,
}

impl Design {
    /// Decode and clamp a MODE vector.
    pub fn decode(x: &[f64]) -> Result<Self, String> {
        if x.len() != DIM {
            return Err(format!("expected {DIM} decision values, got {}", x.len()));
        }
        let bounded = std::array::from_fn::<_, DIM, _>(|index| {
            if x[index].is_finite() {
                x[index].clamp(LOWER[index], UPPER[index])
            } else {
                0.5 * (LOWER[index] + UPPER[index])
            }
        });
        Ok(Self {
            buffer_capacity: bounded[0].round() as usize,
            speed_a: bounded[1],
            speed_b: bounded[2],
            maintenance_threshold_a: bounded[3],
            maintenance_threshold_b: bounded[4],
            rework_probability: bounded[5],
            dispatch_priority: bounded[6],
            staff_a: bounded[7].round() as usize,
            staff_b: bounded[8].round() as usize,
        })
    }

    pub fn as_vector(self) -> [f64; DIM] {
        [
            self.buffer_capacity as f64,
            self.speed_a,
            self.speed_b,
            self.maintenance_threshold_a,
            self.maintenance_threshold_b,
            self.rework_probability,
            self.dispatch_priority,
            self.staff_a as f64,
            self.staff_b as f64,
        ]
    }
}

impl Default for Design {
    fn default() -> Self {
        Self {
            buffer_capacity: 8,
            speed_a: 1.0,
            speed_b: 1.0,
            maintenance_threshold_a: 0.5,
            maintenance_threshold_b: 0.5,
            rework_probability: 0.8,
            dispatch_priority: 0.25,
            staff_a: 2,
            staff_b: 2,
        }
    }
}

/// Aggregated outputs from one stochastic simulation replication.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Metrics {
    pub arrivals: usize,
    pub shipped: usize,
    pub scrapped: usize,
    pub overflowed: usize,
    pub throughput_per_hour: f64,
    pub mean_lead_time: f64,
    pub mean_wip: f64,
    pub energy: f64,
    pub cost_rate: f64,
}

impl Metrics {
    /// Four minimized MODE objectives.
    pub fn objectives(self) -> [f64; OBJECTIVES] {
        [
            -self.throughput_per_hour,
            self.mean_lead_time,
            self.mean_wip,
            self.cost_rate,
        ]
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct SplitMix64 {
    state: u64,
    spare_normal: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.state)
    }

    fn uniform(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn exponential(&mut self, mean: f64) -> f64 {
        -mean * (1.0 - self.uniform()).ln()
    }

    fn normal(&mut self) -> f64 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        self.spare_normal = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn keyed_uniform(seed: u64, order: u64, stage: u64, visit: u8, lane: u64) -> f64 {
    let value = seed
        ^ order.wrapping_mul(0xD6E8_FEB8_6659_FD93)
        ^ stage.wrapping_mul(0xA076_1D64_78BD_642F)
        ^ (visit as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB)
        ^ lane.wrapping_mul(0x8EBC_6AF0_9C88_C6E3);
    let bits = mix64(value) >> 11;
    (bits as f64 + 0.5) * (1.0 / ((1_u64 << 53) as f64))
}

fn keyed_normal(seed: u64, order: u64, stage: u64, visit: u8) -> f64 {
    let u1 = keyed_uniform(seed, order, stage, visit, 0).max(f64::MIN_POSITIVE);
    let u2 = keyed_uniform(seed, order, stage, visit, 1);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn duration(model_minutes: f64) -> Duration {
    Duration::from_secs_f64(model_minutes.max(1.0e-6))
}

fn model_time<M: Model>(cx: &Context<M>) -> f64 {
    cx.time().duration_since(MonotonicTime::EPOCH).as_secs_f64()
}

#[derive(Clone, Debug, Message, Serialize, Deserialize)]
struct Order {
    id: u64,
    born: f64,
    work_factor: f64,
    quality_risk: f64,
    reworks: u8,
}

#[derive(Clone, Copy, Debug, Message, Serialize, Deserialize)]
enum DepartureKind {
    Shipped,
    Scrapped,
    Overflow,
}

#[derive(Clone, Copy, Debug, Message, Serialize, Deserialize)]
enum MetricEvent {
    Arrival {
        at: f64,
    },
    Departure {
        at: f64,
        born: f64,
        kind: DepartureKind,
    },
    Energy {
        at: f64,
        amount: f64,
    },
}

#[derive(Clone, Debug, Message, Serialize, Deserialize)]
struct Completion {
    order: Order,
    energy: f64,
}

#[derive(Serialize, Deserialize)]
struct Source {
    orders: Output<Order>,
    metrics: Output<MetricEvent>,
    rng: SplitMix64,
    next_id: u64,
    horizon: f64,
    mean_interarrival: f64,
}

#[Model]
impl Source {
    fn new(seed: u64, horizon: f64) -> Self {
        Self {
            orders: Output::default(),
            metrics: Output::default(),
            rng: SplitMix64::new(seed),
            next_id: 0,
            horizon,
            mean_interarrival: 0.72,
        }
    }

    async fn start(&mut self, _: (), cx: &Context<Self>) {
        self.schedule_next(cx);
    }

    fn schedule_next(&mut self, cx: &Context<Self>) {
        let now = model_time(cx);
        let wave = 1.0 + 0.25 * (std::f64::consts::TAU * now / 60.0).sin();
        let delay = self
            .rng
            .exponential(self.mean_interarrival / wave.max(0.25));
        if now + delay <= self.horizon {
            cx.schedule_event(duration(delay), schedulable!(Self::emit), ())
                .expect("positive finite arrival delay");
        }
    }

    #[nexosim(schedulable)]
    async fn emit(&mut self, _: (), cx: &Context<Self>) {
        let at = model_time(cx);
        let work_factor = (0.25 * self.rng.normal() - 0.5 * 0.25_f64.powi(2))
            .exp()
            .clamp(0.45, 2.25);
        let order = Order {
            id: self.next_id,
            born: at,
            work_factor,
            quality_risk: 0.008 + 0.018 * self.rng.uniform(),
            reworks: 0,
        };
        self.next_id += 1;
        self.metrics.send(MetricEvent::Arrival { at }).await;
        self.orders.send(order).await;
        self.schedule_next(cx);
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct MachineConfig {
    stage: u64,
    base_time: f64,
    speed: f64,
    maintenance_threshold: f64,
    staff: usize,
    seed: u64,
}

#[derive(Serialize, Deserialize)]
struct MachineA {
    completed: Output<Order>,
    metrics: Output<MetricEvent>,
    queue: VecDeque<Order>,
    busy: usize,
    dispatch_priority: f64,
    config: MachineConfig,
}

#[Model]
impl MachineA {
    fn new(config: MachineConfig, dispatch_priority: f64) -> Self {
        Self {
            completed: Output::default(),
            metrics: Output::default(),
            queue: VecDeque::new(),
            busy: 0,
            dispatch_priority,
            config,
        }
    }

    async fn arrive(&mut self, order: Order, cx: &Context<Self>) {
        self.queue.push_back(order);
        self.dispatch(cx);
    }

    fn dispatch(&mut self, cx: &Context<Self>) {
        while self.busy < self.config.staff && !self.queue.is_empty() {
            let order = remove_priority(&mut self.queue, self.dispatch_priority);
            let completion = machine_completion(order, self.config);
            let processing_time = completion_time(&completion, self.config);
            self.busy += 1;
            cx.schedule_event(
                duration(processing_time),
                schedulable!(Self::complete),
                completion,
            )
            .expect("positive finite processing time");
        }
    }

    #[nexosim(schedulable)]
    async fn complete(&mut self, completion: Completion, cx: &Context<Self>) {
        self.busy = self.busy.saturating_sub(1);
        self.metrics
            .send(MetricEvent::Energy {
                at: model_time(cx),
                amount: completion.energy,
            })
            .await;
        self.completed.send(completion.order).await;
        self.dispatch(cx);
    }
}

#[derive(Serialize, Deserialize)]
struct FiniteBuffer {
    to_machine_b: Output<Order>,
    metrics: Output<MetricEvent>,
    queue: VecDeque<Order>,
    capacity: usize,
    available_b: usize,
    dispatch_priority: f64,
}

#[Model]
impl FiniteBuffer {
    fn new(capacity: usize, staff_b: usize, dispatch_priority: f64) -> Self {
        Self {
            to_machine_b: Output::default(),
            metrics: Output::default(),
            queue: VecDeque::new(),
            capacity,
            available_b: staff_b,
            dispatch_priority,
        }
    }

    async fn push(&mut self, order: Order, cx: &Context<Self>) {
        if self.available_b > 0 {
            self.available_b -= 1;
            self.to_machine_b.send(order).await;
        } else if self.queue.len() < self.capacity {
            self.queue.push_back(order);
        } else {
            self.metrics
                .send(MetricEvent::Departure {
                    at: model_time(cx),
                    born: order.born,
                    kind: DepartureKind::Overflow,
                })
                .await;
        }
    }

    async fn release(&mut self, _: ()) {
        self.available_b += 1;
        if !self.queue.is_empty() {
            self.available_b -= 1;
            let order = remove_priority(&mut self.queue, self.dispatch_priority);
            self.to_machine_b.send(order).await;
        }
    }
}

#[derive(Serialize, Deserialize)]
struct MachineB {
    completed: Output<Order>,
    release_slot: Output<()>,
    metrics: Output<MetricEvent>,
    config: MachineConfig,
}

#[Model]
impl MachineB {
    fn new(config: MachineConfig) -> Self {
        Self {
            completed: Output::default(),
            release_slot: Output::default(),
            metrics: Output::default(),
            config,
        }
    }

    async fn process(&mut self, order: Order, cx: &Context<Self>) {
        let completion = machine_completion(order, self.config);
        let processing_time = completion_time(&completion, self.config);
        cx.schedule_event(
            duration(processing_time),
            schedulable!(Self::complete),
            completion,
        )
        .expect("positive finite processing time");
    }

    #[nexosim(schedulable)]
    async fn complete(&mut self, completion: Completion, cx: &Context<Self>) {
        self.metrics
            .send(MetricEvent::Energy {
                at: model_time(cx),
                amount: completion.energy,
            })
            .await;
        self.completed.send(completion.order).await;
        self.release_slot.send(()).await;
    }
}

#[derive(Serialize, Deserialize)]
struct Inspection {
    rework: Output<Order>,
    metrics: Output<MetricEvent>,
    rework_probability: f64,
    seed: u64,
}

#[Model]
impl Inspection {
    fn new(rework_probability: f64, seed: u64) -> Self {
        Self {
            rework: Output::default(),
            metrics: Output::default(),
            rework_probability,
            seed,
        }
    }

    async fn inspect(&mut self, mut order: Order, cx: &Context<Self>) {
        let quality_draw = keyed_uniform(self.seed, order.id, 3, order.reworks, 0);
        let failed = quality_draw < order.quality_risk.clamp(0.0, 0.8);
        if !failed {
            self.metrics
                .send(MetricEvent::Departure {
                    at: model_time(cx),
                    born: order.born,
                    kind: DepartureKind::Shipped,
                })
                .await;
            return;
        }

        let route_draw = keyed_uniform(self.seed, order.id, 3, order.reworks, 1);
        if order.reworks < MAX_REWORKS && route_draw < self.rework_probability {
            order.reworks += 1;
            order.quality_risk = (0.45 * order.quality_risk).max(0.003);
            self.rework.send(order).await;
        } else {
            self.metrics
                .send(MetricEvent::Departure {
                    at: model_time(cx),
                    born: order.born,
                    kind: DepartureKind::Scrapped,
                })
                .await;
        }
    }
}

fn remove_priority(queue: &mut VecDeque<Order>, priority: f64) -> Order {
    let len = queue.len();
    let best = queue
        .iter()
        .enumerate()
        .min_by(|(left_index, left), (right_index, right)| {
            let denominator = (len.saturating_sub(1)).max(1) as f64;
            let left_score =
                (1.0 - priority) * (*left_index as f64 / denominator) + priority * left.work_factor;
            let right_score = (1.0 - priority) * (*right_index as f64 / denominator)
                + priority * right.work_factor;
            left_score
                .total_cmp(&right_score)
                .then_with(|| left.id.cmp(&right.id))
        })
        .map(|(index, _)| index)
        .expect("non-empty queue");
    queue.remove(best).expect("valid queue index")
}

fn machine_completion(mut order: Order, config: MachineConfig) -> Completion {
    let interval = (3.0 + 28.0 * config.maintenance_threshold).round().max(2.0) as u64;
    let cycle = order.id.wrapping_add(order.reworks as u64 * 7) % interval;
    let planned_maintenance = cycle == 0;
    let wear = cycle as f64 / interval as f64;
    let variation = (0.16 * keyed_normal(config.seed, order.id, config.stage, order.reworks)).exp();
    let process = config.base_time * order.work_factor * variation / config.speed;
    let failure_probability = (0.008 + 0.055 * wear * wear * config.speed.powi(2)).clamp(0.0, 0.45);
    let failed =
        keyed_uniform(config.seed, order.id, config.stage, order.reworks, 2) < failure_probability;
    let repair = if failed {
        1.5 + 5.0 * -keyed_uniform(config.seed, order.id, config.stage, order.reworks, 3).ln()
    } else {
        0.0
    };
    let maintenance = if planned_maintenance {
        1.2 + 0.9 / config.speed
    } else {
        0.0
    };
    order.quality_risk = (order.quality_risk
        + 0.012 * (config.speed - 1.0).max(0.0)
        + 0.018 * wear
        + if failed { 0.025 } else { 0.0 })
    .clamp(0.0, 0.8);
    let energy = process * (0.55 + config.speed.powi(3)) + 0.30 * repair + 0.20 * maintenance;
    Completion {
        order,
        energy: energy + 1.0e-12 * (repair + maintenance),
    }
}

fn completion_time(completion: &Completion, config: MachineConfig) -> f64 {
    // Recompute the deterministic timing components. Keeping only the order and
    // energy in the scheduled message makes save/restore state compact.
    let order = &completion.order;
    let interval = (3.0 + 28.0 * config.maintenance_threshold).round().max(2.0) as u64;
    let cycle = order.id.wrapping_add(order.reworks as u64 * 7) % interval;
    let wear = cycle as f64 / interval as f64;
    let variation = (0.16 * keyed_normal(config.seed, order.id, config.stage, order.reworks)).exp();
    let process = config.base_time * order.work_factor * variation / config.speed;
    let failure_probability = (0.008 + 0.055 * wear * wear * config.speed.powi(2)).clamp(0.0, 0.45);
    let failed =
        keyed_uniform(config.seed, order.id, config.stage, order.reworks, 2) < failure_probability;
    let repair = if failed {
        1.5 + 5.0 * -keyed_uniform(config.seed, order.id, config.stage, order.reworks, 3).ln()
    } else {
        0.0
    };
    let maintenance = if cycle == 0 {
        1.2 + 0.9 / config.speed
    } else {
        0.0
    };
    process + repair + maintenance
}

/// Run one replication. One `Duration` second represents one model minute.
pub fn simulate(
    design: Design,
    seed: u64,
    horizon_minutes: f64,
    nexosim_threads: usize,
) -> Result<Metrics, String> {
    if !horizon_minutes.is_finite() || horizon_minutes <= 0.0 {
        return Err("horizon must be finite and positive".to_string());
    }
    if nexosim_threads == 0 {
        return Err("NeXosim thread count must be positive".to_string());
    }

    let mut source = Source::new(seed ^ 0xA076_1D64_78BD_642F, horizon_minutes);
    let mut machine_a = MachineA::new(
        MachineConfig {
            stage: 1,
            base_time: 1.45,
            speed: design.speed_a,
            maintenance_threshold: design.maintenance_threshold_a,
            staff: design.staff_a,
            seed: seed ^ 0xE703_7ED1_A0B4_28DB,
        },
        design.dispatch_priority,
    );
    let mut buffer = FiniteBuffer::new(
        design.buffer_capacity,
        design.staff_b,
        design.dispatch_priority,
    );
    let mut machine_b = MachineB::new(MachineConfig {
        stage: 2,
        base_time: 1.75,
        speed: design.speed_b,
        maintenance_threshold: design.maintenance_threshold_b,
        staff: design.staff_b,
        seed: seed ^ 0x8EBC_6AF0_9C88_C6E3,
    });
    let mut inspection = Inspection::new(design.rework_probability, seed ^ 0x5899_65CC_7537_4CC3);

    let source_mailbox = Mailbox::new();
    let machine_a_mailbox = Mailbox::new();
    let buffer_mailbox = Mailbox::new();
    let machine_b_mailbox = Mailbox::new();
    let inspection_mailbox = Mailbox::new();

    source.orders.connect(MachineA::arrive, &machine_a_mailbox);
    machine_a
        .completed
        .connect(FiniteBuffer::push, &buffer_mailbox);
    buffer
        .to_machine_b
        .connect(MachineB::process, &machine_b_mailbox);
    machine_b
        .completed
        .connect(Inspection::inspect, &inspection_mailbox);
    machine_b
        .release_slot
        .connect(FiniteBuffer::release, &buffer_mailbox);
    inspection
        .rework
        .connect(MachineA::arrive, &machine_a_mailbox);

    let (metric_sink, mut metric_reader) = event_queue(SinkState::Enabled);
    source.metrics.connect_sink(metric_sink.clone());
    machine_a.metrics.connect_sink(metric_sink.clone());
    buffer.metrics.connect_sink(metric_sink.clone());
    machine_b.metrics.connect_sink(metric_sink.clone());
    inspection.metrics.connect_sink(metric_sink);

    let mut bench = SimInit::with_num_threads(nexosim_threads);
    let start = EventSource::new()
        .connect(Source::start, &source_mailbox)
        .register(&mut bench);
    let mut simulation = bench
        .add_model(source, source_mailbox, "source")
        .add_model(machine_a, machine_a_mailbox, "machine A")
        .add_model(buffer, buffer_mailbox, "finite buffer")
        .add_model(machine_b, machine_b_mailbox, "machine B")
        .add_model(inspection, inspection_mailbox, "inspection")
        .init(MonotonicTime::EPOCH)
        .map_err(|error| format!("failed to initialize NeXosim bench: {error}"))?;
    simulation
        .process_event(&start, ())
        .map_err(|error| format!("failed to start NeXosim bench: {error}"))?;
    simulation
        .step_until(duration(horizon_minutes))
        .map_err(|error| format!("NeXosim execution failed: {error}"))?;

    let mut events = std::iter::from_fn(|| metric_reader.try_read()).collect::<Vec<_>>();
    events.sort_by(|left, right| event_time(left).total_cmp(&event_time(right)));
    Ok(aggregate(
        &events,
        horizon_minutes,
        design.staff_a + design.staff_b,
    ))
}

fn event_time(event: &MetricEvent) -> f64 {
    match event {
        MetricEvent::Arrival { at }
        | MetricEvent::Departure { at, .. }
        | MetricEvent::Energy { at, .. } => *at,
    }
}

fn aggregate(events: &[MetricEvent], horizon: f64, staff: usize) -> Metrics {
    let mut result = Metrics::default();
    let mut wip = 0_i64;
    let mut wip_area = 0.0;
    let mut last_time = 0.0;
    let mut lead_sum = 0.0;
    for event in events {
        let at = event_time(event).clamp(last_time, horizon);
        wip_area += wip.max(0) as f64 * (at - last_time);
        last_time = at;
        match *event {
            MetricEvent::Arrival { .. } => {
                result.arrivals += 1;
                wip += 1;
            }
            MetricEvent::Departure { at, born, kind, .. } => {
                wip -= 1;
                match kind {
                    DepartureKind::Shipped => {
                        result.shipped += 1;
                        lead_sum += at - born;
                    }
                    DepartureKind::Scrapped => result.scrapped += 1,
                    DepartureKind::Overflow => result.overflowed += 1,
                }
            }
            MetricEvent::Energy { amount, .. } => result.energy += amount,
        }
    }
    wip_area += wip.max(0) as f64 * (horizon - last_time);
    result.throughput_per_hour = 60.0 * result.shipped as f64 / horizon;
    result.mean_lead_time = if result.shipped > 0 {
        lead_sum / result.shipped as f64
    } else {
        2.0 * horizon
    };
    result.mean_wip = wip_area / horizon;
    result.cost_rate = result.energy / horizon + 0.65 * staff as f64;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn design_decode_clamps_and_rounds() {
        let design =
            Design::decode(&[-10.0, 2.0, f64::NAN, 0.0, 2.0, -1.0, 2.0, 2.6, 9.0]).unwrap();
        assert_eq!(design.buffer_capacity, 1);
        assert_eq!(design.staff_a, 3);
        assert_eq!(design.staff_b, 4);
        assert_eq!(design.speed_a, UPPER[1]);
        assert_eq!(design.speed_b, 0.5 * (LOWER[2] + UPPER[2]));
        assert_eq!(design.rework_probability, 0.0);
        assert_eq!(design.dispatch_priority, 1.0);
        assert!(Design::decode(&[0.0; DIM - 1]).is_err());
    }

    #[test]
    fn simulation_is_finite_and_reproducible() {
        let first = simulate(Design::default(), 7, 30.0, 1).unwrap();
        let second = simulate(Design::default(), 7, 30.0, 1).unwrap();
        assert_eq!(first, second);
        assert!(first.arrivals > 0);
        assert!(first.objectives().into_iter().all(f64::is_finite));
    }

    #[test]
    fn executor_width_preserves_seeded_result() {
        let serial = simulate(Design::default(), 11, 30.0, 1).unwrap();
        let parallel = simulate(Design::default(), 11, 30.0, 4).unwrap();
        assert_eq!(serial, parallel);
    }

    #[test]
    fn invalid_simulation_configuration_is_rejected() {
        assert!(simulate(Design::default(), 1, 0.0, 1).is_err());
        assert!(simulate(Design::default(), 1, 10.0, 0).is_err());
    }
}
