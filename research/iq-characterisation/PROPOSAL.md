# Proposal: characterise captured IQ

Label what a recording from `run`/`capture` looks like, as a **heuristic hint with evidence**, never as proof and never touching the Artemis identification.

## What the research showed (synthetic signals only; see FINDINGS.md)

**Does not work as imagined**
- RF-Sentinel-style envelope kurtosis of the raw capture is blind to narrowband signals: a 12.5 kHz signal is ~0.6 % of a 2 Msps capture, so the value stays at the noise level (3.24) even at 30 dB. It cannot separate a carrier from noise.
- Kurtosis only helps after finding the signal's band, subtracting the noise, and using the active part only. Then it gives a modulation-class value (constant envelope 1.00, QPSK 1.20, BPSK 1.37, OFDM 1.99, same as noise). It is not a "bursty" test; duty and burst statistics do that.
- "Wideband noise" cannot be told from a noise-like signal (OFDM, Wi-Fi). The honest labels are "no signal in capture" and "wideband, noise-like".

**Works well on the synthetic bench**
- Noise-only vs any structure: spectral excess over a robust floor, ~0.2 % false alarms on noise captures, detects CW down to about -12 dB.
- Occupied bandwidth (99 %): narrow / medium / wide.
- Single carrier: one narrow spectral line. It cannot tell CW from AM or an unmodulated FM carrier.
- On/off structure: duty and burst lengths from a noise-referenced model (duty error <= 0.02 from 0 dB for packets).
- OOK-like: short, quantised on/off runs plus a chip-rate hint (within 10 % at >= 10 dB in-band SNR). Regular non-OOK timing can also score high.
- OFDM-like via cyclic-prefix test (0 false positives in ~5,700 non-OFDM captures); frequency hopping with dwell time.

**Not measured (the run ran out before these)**: baud rate for FSK/PSK, LoRa/chirp detection, and the final combined classifier's accuracy. Nothing was tested on real HackRF captures.

## Labels (closed list, always including "unknown")

no signal in capture - single carrier - keyed on/off (OOK-like, chip-rate hint) - bursty packets - continuous narrow/medium (FM/FSK-like, **no** baud claim) - wideband flat / OFDM-like - hopping - wideband noise-like - unknown.
Shown as low/medium/high plus the numbers behind it, not as a percentage. Quality flags cap trust: mostly silent capture, low SNR, ADC clipping, DC spike.

## Where it goes

- New numpy-only module `hackrf_scout/iqchar.py`, two streaming passes over the `.sigmf-data`. Measured on x86: ~5 s CPU and ~215 MB for 5 s at 10 Msps (100 MB); expect 60-90 s on a Raspberry Pi. 2 Msps captures take 1-2 s.
- Hook: in `_do_captures` after the capture row is saved (cli.py:349-351), wrapped so an analysis error never loses a capture. Plus `hackrf-scout characterise FILE.sigmf-meta | --signal N | --all` (no HackRF, no lock) for existing captures.
- Storage: new columns on `captures` (label, confidence word, quality flags, features JSON, method version). Edited straight into the schema (your no-compatibility rule), so an existing `scout.db` must be recreated or the first insert fails.
- Show: `report`, the captures list and the signal drawer, with a neutral "heuristic" tag.

## Plan

1. Core: floor and band finding, noise-only, carrier, bandwidth, duty/bursts, quality flags, tests on synthetic IQ.
2. OOK chip rate, OFDM, hopping.
3. FSK/PSK baud hint and chirp, only once measured.
4. Validate on real captures (your 88 MB files plus ~20 hand-labelled ones) before wording the labels as anything more than hints.

## Decisions for you

- Run inline after each capture (delays the next scan by seconds, up to ~90 s on a Pi) or only via `characterise`?
- Start with phase 1 only?
