//! Compares tract runtimes on the per-frame path that libDF actually runs.
//!
//! The unit of work is one `hop_size` frame, because that -- not whole-file
//! throughput -- is what the LADSPA plugin and the live demo are latency-bound
//! on. Tail latency is reported alongside the median: a backend that misses its
//! deadline on every tenth frame is unusable even with a good mean.
//!
//!     cargo bench -p deep_filter --features tract,default-model --bench runtime
//!     cargo bench -p deep_filter --features tract,default-model,gpu-metal --bench runtime
//!
//! `DF_RUNTIME` is deliberately not set here; each case names its runtime
//! explicitly, but note that the env var still overrides it, so leave it unset.

use std::time::{Duration, Instant};

use df::tract::*;
use ndarray::Array2;

const WARMUP_FRAMES: usize = 200;
const MEASURED_FRAMES: usize = 2000;

/// Deterministic noise at a fixed RMS. It has to sit above `min_db_thresh`, or
/// `apply_stages` gates the decoders off and the benchmark measures the encoder
/// alone; well below full scale, or `process` clamps.
fn noise(n_ch: usize, hop_size: usize, seed: &mut u32) -> Array2<f32> {
    Array2::from_shape_fn((n_ch, hop_size), |_| {
        // xorshift32: no dev-dependency, and identical across runs and hosts.
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        (*seed as f32 / u32::MAX as f32 - 0.5) * 0.2
    })
}

struct Stats {
    build: Duration,
    p50: Duration,
    p90: Duration,
    p99: Duration,
    max: Duration,
    mean: Duration,
}

impl Stats {
    /// Fraction of a frame's wall-clock budget spent computing it. Below 1.0 is
    /// the bare minimum for real-time use.
    fn rtf(&self, hop_size: usize, sr: usize) -> f32 {
        self.mean.as_secs_f32() / (hop_size as f32 / sr as f32)
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn measure(runtime: &str, n_ch: usize) -> anyhow::Result<(Stats, usize, usize, Vec<f32>)> {
    let params = RuntimeParams::default_with_ch(n_ch).with_runtime(runtime);

    let t0 = Instant::now();
    let mut model = DfTract::new(DfParams::default(), &params)?;
    let build = t0.elapsed();

    let (hop_size, sr) = (model.hop_size, model.sr);
    let mut seed = 0x1234_5678u32;
    let mut enh = Array2::<f32>::zeros((n_ch, hop_size));

    for _ in 0..WARMUP_FRAMES {
        let noisy = noise(n_ch, hop_size, &mut seed);
        model.process(noisy.view(), enh.view_mut())?;
    }

    // Retained so a GPU backend can be checked against the CPU for agreement,
    // not just for speed. A fast wrong answer is not a result.
    let mut output = Vec::with_capacity(MEASURED_FRAMES * n_ch * hop_size);
    let mut samples = Vec::with_capacity(MEASURED_FRAMES);
    for _ in 0..MEASURED_FRAMES {
        let noisy = noise(n_ch, hop_size, &mut seed);
        let t = Instant::now();
        model.process(noisy.view(), enh.view_mut())?;
        samples.push(t.elapsed());
        output.extend(enh.iter().copied());
    }

    let total: Duration = samples.iter().sum();
    samples.sort_unstable();
    let stats = Stats {
        build,
        p50: percentile(&samples, 0.50),
        p90: percentile(&samples, 0.90),
        p99: percentile(&samples, 0.99),
        max: samples[samples.len() - 1],
        mean: total / samples.len() as u32,
    };
    Ok((stats, hop_size, sr, output))
}

fn main() -> anyhow::Result<()> {
    let _ = env_logger::try_init();

    let registered: Vec<String> =
        tract_core::prelude::runtimes().map(|rt| rt.name().to_string()).collect();
    println!("registered tract backends: {registered:?}");

    // Only ask for a GPU when one is compiled in. Otherwise `gpu-or-cpu` would
    // resolve straight back to the CPU and we would benchmark it twice.
    let mut cases: Vec<&str> = vec!["cpu"];
    if cfg!(any(feature = "gpu-metal", feature = "gpu-cuda")) {
        cases.push("gpu-or-cpu");
    } else {
        println!("no GPU backend compiled in; build with --features gpu-metal or gpu-cuda");
    }

    println!(
        "\n{:>12} {:>3} {:>9} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "runtime", "ch", "build", "p50", "p90", "p99", "max", "RTF"
    );

    for n_ch in [1usize, 2] {
        let mut reference: Option<Vec<f32>> = None;
        for runtime in &cases {
            let (s, hop_size, sr, out) = measure(runtime, n_ch)?;
            println!(
                "{:>12} {:>3} {:>9.2?} {:>10.3?} {:>10.3?} {:>10.3?} {:>10.3?} {:>8.4}",
                runtime,
                n_ch,
                s.build,
                s.p50,
                s.p90,
                s.p99,
                s.max,
                s.rtf(hop_size, sr)
            );

            // Same seed and frame count across runtimes, so outputs of one
            // channel count are directly comparable.
            match &reference {
                None => reference = Some(out),
                Some(reference) => {
                    let worst =
                        reference.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    let verdict = if worst < 1e-3 { "ok" } else { "MISMATCH" };
                    println!(
                        "{:>12} {:>3}   max abs deviation from cpu: {worst:.3e}  {verdict}",
                        "", n_ch
                    );
                }
            }
        }
    }
    Ok(())
}
