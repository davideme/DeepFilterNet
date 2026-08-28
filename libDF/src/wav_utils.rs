use std::result::Result;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read},
};

use hound::{WavReader, WavWriter};
#[cfg(any(feature = "dataset", feature = "wav-utils"))]
use ndarray::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WavUtilsError {
    #[error("Hound Error")]
    HoundError(#[from] hound::Error),
    #[error("Hound Error Detail")]
    HoundErrorDetail { source: hound::Error, msg: String },
    #[error("Ndarray Shape Error")]
    NdarrayShapeError(#[from] ndarray::ShapeError),
}

pub struct ReadWav {
    reader: WavReader<BufReader<File>>,
    pub channels: usize,
    pub sr: usize,
    pub len: usize,
    pub dtype: hound::SampleFormat,
}

impl ReadWav {
    pub fn new(path: &str) -> Result<Self, WavUtilsError>
    where
        Self: Sized,
    {
        let reader = match WavReader::open(path) {
            Err(e) => {
                return Err(WavUtilsError::HoundErrorDetail {
                    source: e,
                    msg: format!("Could not find audio file {path}"),
                })
            }
            Ok(r) => r,
        };
        let spec = reader.spec();
        let channels = spec.channels as usize;
        let sr = spec.sample_rate as usize;
        let len = reader.len() as usize / channels;
        let dtype = spec.sample_format;
        Ok(ReadWav {
            reader,
            channels,
            sr,
            len,
            dtype,
        })
    }
    pub fn iter(&mut self) -> Box<dyn Iterator<Item = f32> + '_> {
        match self.dtype {
            hound::SampleFormat::Int => Box::new(read_wav_raw_i16(&mut self.reader)),
            hound::SampleFormat::Float => Box::new(read_wav_raw_f32(&mut self.reader)),
        }
    }
    pub fn samples_vec(mut self) -> Result<Vec<Vec<f32>>, WavUtilsError> {
        let mut out = vec![Vec::<f32>::new(); self.channels];
        let mut samples = self.iter();
        'outer: loop {
            for out_ch in out.iter_mut() {
                match samples.next() {
                    None => break 'outer,
                    Some(x) => out_ch.push(x),
                }
            }
        }
        Ok(out)
    }
    #[cfg(any(feature = "dataset", feature = "wav-utils"))]
    pub fn samples_arr2(mut self) -> Result<Array2<f32>, WavUtilsError> {
        Ok(
            Array2::from_shape_vec((self.len, self.channels), self.iter().collect())?
                .t()
                .to_owned(),
        )
    }
    /// Read up to `max_frames` frames into `out[ch][..n]`, deinterleaved.
    ///
    /// Returns the number of frames read. A return value below `max_frames` signals EOF. `out`
    /// is grown to `max_frames` per channel if needed; samples beyond the returned count are
    /// left untouched and must not be read.
    ///
    /// Unlike [`ReadWav::iter`], decoding errors are propagated rather than panicking, which
    /// matters when streaming long or truncated files.
    pub fn read_frames(
        &mut self,
        out: &mut [Vec<f32>],
        max_frames: usize,
    ) -> Result<usize, WavUtilsError> {
        debug_assert_eq!(out.len(), self.channels);
        for ch in out.iter_mut() {
            if ch.len() < max_frames {
                ch.resize(max_frames, 0.);
            }
        }
        let channels = self.channels;
        match self.dtype {
            hound::SampleFormat::Int => {
                read_frames_into(&mut self.reader, out, max_frames, channels, |s: i16| {
                    s as f32 / 32767.0
                })
            }
            hound::SampleFormat::Float => {
                read_frames_into(&mut self.reader, out, max_frames, channels, |s: f32| s)
            }
        }
    }
}

/// Deinterleave up to `max_frames` frames of `S` samples into `out`, converting via `conv`.
///
/// Stops early on EOF; a partially decoded trailing frame is discarded.
fn read_frames_into<S, R, F>(
    reader: &mut WavReader<R>,
    out: &mut [Vec<f32>],
    max_frames: usize,
    channels: usize,
    conv: F,
) -> Result<usize, WavUtilsError>
where
    S: hound::Sample,
    R: Read,
    F: Fn(S) -> f32,
{
    let mut samples = reader.samples::<S>();
    let mut n = 0;
    'outer: while n < max_frames {
        for ch in out.iter_mut().take(channels) {
            match samples.next() {
                None => break 'outer,
                Some(s) => ch[n] = conv(s?),
            }
        }
        n += 1;
    }
    Ok(n)
}

fn read_wav_raw_i16<R: Read>(reader: &mut WavReader<R>) -> impl Iterator<Item = f32> + '_ {
    reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32767.0)
}
fn read_wav_raw_f32<R: Read>(reader: &mut WavReader<R>) -> impl Iterator<Item = f32> + '_ {
    reader.samples::<f32>().map(|s| s.unwrap())
}

pub fn read_wav(path: &str) -> Result<(Vec<Vec<f32>>, u32), WavUtilsError> {
    let mut reader = WavReader::open(path)?;
    let ch = reader.spec().channels as usize;
    let sr = reader.spec().sample_rate;
    let mut out = vec![Vec::<f32>::new(); ch];
    let mut samples = read_wav_raw_i16(&mut reader);
    'outer: loop {
        for out_ch in out.iter_mut() {
            match samples.next() {
                None => break 'outer,
                Some(x) => out_ch.push(x),
            }
        }
    }
    Ok((out, sr))
}

pub fn write_wav_iter<'a, I>(path: &str, iter: I, sr: u32, ch: u16) -> Result<(), WavUtilsError>
where
    I: IntoIterator<Item = &'a f32>,
{
    let spec = hound::WavSpec {
        channels: ch,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;

    for &sample in iter.into_iter() {
        writer.write_sample((sample * i16::MAX as f32) as i16)?;
    }
    Ok(writer.finalize()?)
}

pub fn write_wav(path: &str, x: &[Vec<f32>], sr: u32) -> Result<(), WavUtilsError> {
    let spec = hound::WavSpec {
        channels: x.len() as u16,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;

    for t in 0..x[0].len() {
        for ch in x.iter() {
            writer.write_sample((ch[t] * i16::MAX as f32) as i16)?;
        }
    }
    Ok(writer.finalize()?)
}

#[cfg(any(feature = "dataset", feature = "wav-utils"))]
pub fn write_wav_arr2(path: &str, x: ArrayView2<f32>, sr: u32) -> Result<(), WavUtilsError> {
    let spec = hound::WavSpec {
        channels: x.len_of(Axis(0)) as u16,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for xt in x.axis_iter(Axis(1)) {
        for s in xt.iter() {
            writer.write_sample((s * i16::MAX as f32) as i16)?;
        }
    }
    Ok(writer.finalize()?)
}

/// Incremental 16-bit PCM wav writer.
///
/// Counterpart of [`write_wav_arr2`] for producers that emit audio in blocks rather than as one
/// array. The sample conversion is deliberately identical to [`write_wav_arr2`] so that streamed
/// output is byte-for-byte comparable with the in-memory path.
pub struct WavWriterStream {
    writer: WavWriter<BufWriter<File>>,
    channels: usize,
}

impl WavWriterStream {
    pub fn create(path: &str, sr: u32, channels: u16) -> Result<Self, WavUtilsError> {
        let spec = hound::WavSpec {
            channels,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        Ok(WavWriterStream {
            writer: WavWriter::create(path, spec)?,
            channels: channels as usize,
        })
    }
    /// Write the first `n` frames of a channel-major buffer, interleaving them.
    pub fn write_deinterleaved(
        &mut self,
        chans: &[Vec<f32>],
        n: usize,
    ) -> Result<(), WavUtilsError> {
        debug_assert_eq!(chans.len(), self.channels);
        for t in 0..n {
            for ch in chans.iter() {
                self.writer.write_sample((ch[t] * i16::MAX as f32) as i16)?;
            }
        }
        Ok(())
    }
    pub fn frames_written(&self) -> u32 {
        self.writer.len() / self.channels as u32
    }
    /// Checkpoint the header so an interrupted run still leaves a playable file.
    pub fn flush(&mut self) -> Result<(), WavUtilsError> {
        Ok(self.writer.flush()?)
    }
    pub fn finalize(self) -> Result<(), WavUtilsError> {
        Ok(self.writer.finalize()?)
    }
}
