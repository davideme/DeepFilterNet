use std::{
    path::PathBuf,
    process::exit,
    time::{Duration, Instant},
};

use anyhow::Result;
use clap::{Parser, ValueHint};
use df::{tract::*, transforms::StreamingResampler, wav_utils::*};
use ndarray::prelude::*;

#[cfg(all(
    not(windows),
    not(target_os = "android"),
    not(target_os = "macos"),
    not(target_os = "freebsd"),
    not(target_env = "musl"),
    not(target_arch = "riscv64"),
    feature = "use-jemalloc"
))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// Simple program to sample from a hd5 dataset directory
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to model tar.gz
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    model: Option<PathBuf>,
    /// Enable post-filter
    #[arg(long = "pf")]
    post_filter: bool,
    /// Post-filter beta. Higher beta results in stronger attenuation.
    #[arg(long = "pf-beta", default_value_t = 0.02)]
    post_filter_beta: f32,
    /// Compensate delay of STFT and model lookahead
    #[arg(short = 'D', long)]
    compensate_delay: bool,
    /// Attenuation limit in dB by mixing the enhanced signal with the noisy signal.
    /// An attenuation limit of 0 dB means no noise reduction will be performed, 100 dB means full
    /// noise reduction, i.e. no attenuation limit.
    #[arg(short, long, default_value_t = 100.)]
    atten_lim_db: f32,
    /// Min dB local SNR threshold for running the decoder DNN side
    #[arg(long, value_parser, allow_negative_numbers = true, default_value_t = -15.)]
    min_db_thresh: f32,
    /// Max dB local SNR threshold for running ERB decoder
    #[arg(
        long,
        value_parser,
        allow_negative_numbers = true,
        default_value_t = 35.
    )]
    max_db_erb_thresh: f32,
    /// Max dB local SNR threshold for running DF decoder
    #[arg(
        long,
        value_parser,
        allow_negative_numbers = true,
        default_value_t = 35.
    )]
    max_db_df_thresh: f32,
    /// If used with multiple channels, reduce the mask with max (1) or mean (2)
    #[arg(long, value_parser, default_value_t = 1)]
    reduce_mask: i32,
    /// Logging verbosity
    #[arg(
        long,
        short = 'v',
        action = clap::ArgAction::Count,
        global = true,
        help = "Increase logging verbosity with multiple `-vv`",
    )]
    verbose: u8,
    // Output directory with enhanced audio files. Defaults to 'out'
    #[arg(short, long, default_value = "out", value_hint = ValueHint::DirPath)]
    output_dir: PathBuf,
    // Audio files
    #[arg(required = true)]
    files: Vec<PathBuf>,
}

/// Frames read from disk per iteration. Large enough to amortize IO and the per-call
/// resampler overhead, small enough that memory stays independent of file length.
const READ_FRAMES: usize = 4096;

fn main() -> Result<()> {
    let args = Args::parse();

    let level = match args.verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    let tract_level = match args.verbose {
        0..=3 => log::LevelFilter::Error,
        4 => log::LevelFilter::Info,
        5 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    env_logger::Builder::from_env(env_logger::Env::default())
        .filter_level(level)
        .filter_module("tract_onnx", tract_level)
        .filter_module("tract_hir", tract_level)
        .filter_module("tract_core", tract_level)
        .filter_module("tract_linalg", tract_level)
        .init();

    // Initialize with 1 channel
    let mut r_params = RuntimeParams::default();
    r_params = r_params.with_atten_lim(args.atten_lim_db).with_thresholds(
        args.min_db_thresh,
        args.max_db_erb_thresh,
        args.max_db_df_thresh,
    );
    if args.post_filter {
        r_params = r_params.with_post_filter(args.post_filter_beta);
    }
    if let Ok(red) = args.reduce_mask.try_into() {
        r_params = r_params.with_mask_reduce(red);
    } else {
        log::warn!("Input not valid for `reduce_mask`.")
    }
    let df_params = if let Some(tar) = args.model.as_ref() {
        match DfParams::new(tar.clone()) {
            Ok(p) => p,
            Err(e) => {
                log::error!("Error opening model {}: {}", tar.display(), e);
                exit(1)
            }
        }
    } else if cfg!(any(feature = "default-model", feature = "default-model-ll")) {
        DfParams::default()
    } else {
        log::error!("deep-filter was not compiled with a default model. Please provide a model via '--model <path-to-model.tar.gz>'");
        exit(2)
    };
    // Cloning a pristine template gives each file a fresh model state (STFT overlap,
    // exponential norm state, tract GRU/delay op-states) without re-parsing the ONNX graphs:
    // the plan is behind an `Arc`, only the op-states are deep-copied.
    let mut model_template: DfTract = DfTract::new(df_params.clone(), &r_params)?;
    let mut sr = model_template.sr;
    let mut delay = model_template.fft_size - model_template.hop_size; // STFT delay
    delay += model_template.lookahead * model_template.hop_size; // Add model latency due to lookahead
    if !args.output_dir.is_dir() {
        log::info!("Creating output directory: {}", args.output_dir.display());
        std::fs::create_dir_all(args.output_dir.clone())?
    }
    for file in args.files {
        let mut reader = ReadWav::new(file.to_str().unwrap())?;
        // Check if we need to adjust to multiple channels
        if r_params.n_ch != reader.channels {
            r_params.n_ch = reader.channels;
            model_template = DfTract::new(df_params.clone(), &r_params)?;
            sr = model_template.sr;
        }
        let mut model = model_template.clone();
        let sample_sr = reader.sr;
        let n_ch = reader.channels;
        let hop = model.hop_size;

        // Length budget, computed up front so the writer can be capped exactly. The double
        // ceiling reproduces the two-step resampling of the previous in-memory path.
        let n_in = reader.len;
        let n_model = if sr != sample_sr {
            StreamingResampler::out_len(n_in, sample_sr, sr)
        } else {
            n_in
        };
        // TODO: feeding `ceil(delay / hop)` trailing zero hops would recover the last `delay`
        // samples that `--compensate-delay` currently discards, and make the output the same
        // length as the input. That changes `-D` output, so it is deliberately left out here.
        let skip = if args.compensate_delay { delay } else { 0 };
        if skip > 0 && n_model <= skip {
            log::warn!(
                "File {} is shorter than the model delay ({} samples); writing empty output.",
                file.display(),
                delay
            );
        }
        let n_enh = n_model.saturating_sub(skip);
        let n_out = if sr != sample_sr {
            StreamingResampler::out_len(n_enh, sr, sample_sr)
        } else {
            n_enh
        };

        let mut enh_file = args.output_dir.clone();
        enh_file.push(file.file_name().unwrap());
        let mut writer =
            WavWriterStream::create(enh_file.to_str().unwrap(), sample_sr as u32, n_ch as u16)?;
        // Both are zero-cost passthroughs when the rates already match.
        let mut in_rs = StreamingResampler::new(sample_sr, sr, n_ch, None)?;
        let mut out_rs = StreamingResampler::new(sr, sample_sr, n_ch, None)?;

        // All fixed capacity: the queues never hold more than one read block plus one
        // resampler chunk plus one hop, so memory is independent of the file length.
        let mut in_ch: Vec<Vec<f32>> = vec![vec![0.; READ_FRAMES]; n_ch];
        let mut model_q: Vec<Vec<f32>> = vec![Vec::new(); n_ch];
        let mut enh_q: Vec<Vec<f32>> = vec![Vec::new(); n_ch];
        let mut out_q: Vec<Vec<f32>> = vec![Vec::new(); n_ch];
        // Owned and standard layout: `DfTract::process` requires contiguous per-channel rows.
        let mut noisy_f: Array2<f32> = Array2::zeros((n_ch, hop));
        let mut enh_f: Array2<f32> = Array2::zeros((n_ch, hop));

        let mut skip_remaining = skip;
        let mut written = 0usize;
        let mut frames_done = 0usize;
        let mut last_flush = 0usize;
        let t0 = Instant::now();
        let mut last_log = t0;
        let t_audio = n_model as f32 / sr as f32;
        let log_progress = t_audio > 60.;

        loop {
            let n = reader.read_frames(&mut in_ch, READ_FRAMES)?;
            let eof = n < READ_FRAMES;
            {
                let rows: Vec<&[f32]> = in_ch.iter().map(|c| c.as_slice()).collect();
                in_rs.push(&rows, n, &mut model_q)?;
            }
            if eof {
                in_rs.flush(&mut model_q)?;
            }

            // Feed the model one hop at a time. At EOF the final partial hop is zero-padded and
            // still processed; only its first `k` output samples are kept.
            let avail = model_q[0].len();
            let mut consumed = 0;
            while avail - consumed >= hop || (eof && consumed < avail) {
                let k = (avail - consumed).min(hop);
                {
                    let noisy_s = noisy_f.as_slice_mut().unwrap();
                    for (ch, q) in model_q.iter().enumerate() {
                        let row = &mut noisy_s[ch * hop..(ch + 1) * hop];
                        row[..k].copy_from_slice(&q[consumed..consumed + k]);
                        row[k..].fill(0.);
                    }
                }
                model.process(noisy_f.view(), enh_f.view_mut())?;
                {
                    let enh_s = enh_f.as_slice().unwrap();
                    for (ch, q) in enh_q.iter_mut().enumerate() {
                        q.extend_from_slice(&enh_s[ch * hop..ch * hop + k]);
                    }
                }
                consumed += k;
                frames_done += k;
            }
            for q in model_q.iter_mut() {
                q.drain(..consumed);
            }

            // Delay compensation is a leading-sample skip, applied before the output resampler
            // to match the order of the previous in-memory path.
            if skip_remaining > 0 {
                let s = skip_remaining.min(enh_q[0].len());
                for q in enh_q.iter_mut() {
                    q.drain(..s);
                }
                skip_remaining -= s;
            }
            {
                let m = enh_q[0].len();
                let rows: Vec<&[f32]> = enh_q.iter().map(|c| c.as_slice()).collect();
                out_rs.push(&rows, m, &mut out_q)?;
            }
            for q in enh_q.iter_mut() {
                q.clear();
            }
            if eof {
                out_rs.flush(&mut out_q)?;
            }

            let w = out_q[0].len().min(n_out.saturating_sub(written));
            writer.write_deinterleaved(&out_q, w)?;
            written += w;
            for q in out_q.iter_mut() {
                q.clear();
            }

            if eof {
                break;
            }
            if log_progress && last_log.elapsed() >= Duration::from_secs(10) {
                let t_done = frames_done as f32 / sr as f32;
                let elapsed = t0.elapsed().as_secs_f32();
                let rtf = elapsed / t_done;
                log::info!(
                    "{}: {:.0}/{:.0}s ({:.1}%), RTF: {:.3}, ETA: {:.0}s",
                    file.display(),
                    t_done,
                    t_audio,
                    100. * t_done / t_audio,
                    rtf,
                    (t_audio - t_done) * rtf,
                );
                last_log = Instant::now();
            }
            // Checkpoint the header every ~30s of audio so an interrupted run stays playable.
            if frames_done - last_flush >= sr * 30 {
                writer.flush()?;
                last_flush = frames_done;
            }
        }

        if written < n_out {
            log::warn!(
                "File {} ended early: padding {} samples of silence.",
                file.display(),
                n_out - written
            );
            let zeros: Vec<Vec<f32>> = vec![vec![0.; READ_FRAMES]; n_ch];
            while written < n_out {
                let w = READ_FRAMES.min(n_out - written);
                writer.write_deinterleaved(&zeros, w)?;
                written += w;
            }
        }
        debug_assert_eq!(written, n_out);
        writer.finalize()?;

        let elapsed = t0.elapsed().as_secs_f32();
        log::info!(
            "Enhanced audio file {} in {:.2} (RTF: {})",
            file.display(),
            elapsed,
            elapsed / t_audio
        );
    }

    Ok(())
}
