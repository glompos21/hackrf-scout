# hackrf-scout

Sweep the whole HackRF range (1 MHz – 6 GHz), find signals, work out what they probably are, and keep everything in a SQLite database. Optionally record a short IQ capture of each new signal.

```
hackrf_sweep ──► detect (noise floor + SNR + persistence) ──► identify (Artemis/SigID Wiki + bandplan) ──► SQLite
                                                                                                    └──► IQ captures (.cs8 + .json)
```

## Install

```bash
sudo apt install hackrf          # provides hackrf_sweep and hackrf_transfer
pip install -e .                 # needs Python 3.9+ and numpy
hackrf-scout update-db           # one-off: downloads the SigID Wiki database (~300 MB), keeps ~1 MB of JSON
```

`update-db` downloads the Artemis-DB release, verifies its SHA-256 against the published value, opens it read-only and writes `data/artemis_signals.json`. The tool works without it, using only the built-in bandplan (EU / ITU Region 1), but names will be much more generic.

## Use

```bash
# Scan everything, keep scanning until Ctrl-C. New signals are printed as they are confirmed.
hackrf-scout scan --region-keywords greece,cyprus

# Only some ranges, with the RF amp on (weak signals)
hackrf-scout scan -f 400:470 -f 860:960 -a

# Scan 10 minutes, then record 5 s of IQ from the best new signals, repeat forever
hackrf-scout run --scan-seconds 600 --capture-seconds 5 --capture-max 5

# Look at what was found
hackrf-scout report                     # everything
hackrf-scout report --unidentified -v   # the interesting ones, with candidate matches
hackrf-scout report --band 430:440 --sort max_snr
hackrf-scout export --format csv > signals.csv
hackrf-scout export --format json --observations > signals.json

# Re-run identification after updating the database or changing the region
hackrf-scout identify --all --region-keywords greece
```

A signal is stored only after it shows up in `--min-hits` sweeps (default 3). Single-sweep blips, which are the usual false positives, are discarded. Re-detecting a known signal updates its row (hits, max SNR, last seen) and adds a time-stamped observation at most every `--obs-interval` seconds.

### Try it without a HackRF

```bash
hackrf-scout simulate -f 1:3000 -N 12 > sim.csv
hackrf-scout scan --source sim.csv --db demo.db
hackrf-scout report --db demo.db
```

## How identification works

Each stored signal is scored against every Artemis entry: **frequency 40 + bandwidth 30 + region 10**, shown as a percentage. The best entry becomes the name when it scores at least `--min-score` (default 55). Two details matter:

* The database does not say whether a multi-value frequency entry is a span or a list of channels. Matches from wide ranges (over 5 MHz) are flagged `broad`. When the bandplan already knows what lives there (FM broadcast, GSM, ADS-B, Wi-Fi …) the bandplan name is used and the broad Artemis hits are shown only as candidates (`report -v`).
* **Modulation is not scored.** A sweep only measures power, so the tool cannot tell FSK from OOK. The candidates list their usual modulation as a hint; confirm it by hand with a demodulator (SDR++, URH, inspectrum) using the saved IQ.

Treat names as leads, not facts. Many consumer devices (key fobs, sensors) share 433/868 MHz, so a single-point match there is often one of several equally good candidates; read the list with `report -v`.

## Tuning

| Symptom | Try |
|---|---|
| Too many weak/false detections | raise `--snr` (12–15) or `--min-hits` (5) |
| Missing weak signals | lower `--snr` (7–8), add `-a`, or scan narrower ranges so each sweep is faster |
| Overload / ghost signals near strong transmitters | lower `-l`/`-g`, leave `-a` off, use `--ignore` for known spurs |
| Short bursts (LoRa, remotes, TPMS) missed | sweeps are about 1–3 s over the whole range; scan only the relevant bands (`-f 433:435`) for much faster revisits |
| Frequency resolution | `-w 100000` is the default; smaller bins resolve narrow signals better but slow the sweep |

## Limits

* Power values are relative dB from `hackrf_sweep`, not calibrated dBm.
* `hackrf_sweep` and `hackrf_transfer` cannot use the HackRF at the same time, so IQ capture only happens between scan cycles (`run`). Signals that are not transmitting during the capture window produce silent recordings.
* The HackRF has 8-bit ADC and image/spur products. Very strong signals produce ghost copies; check anything suspicious with a lower gain.
* Only what is on the air during the scan is seen. Intermittent signals need long runs.
* The simulator tests the pipeline, not real-world false-positive rates. Expect to tune `--snr`, `--min-hits` and `--ignore` for your location.

## Files

* `scout.db`: SQLite with tables `signals`, `observations`, `sweeps`, `captures`, `meta`.
* `captures/`: `sigNNNN_<freq>MHz_<time>.cs8` (signed 8-bit interleaved IQ) plus a `.json` sidecar with centre frequency, sample rate and gains.

## Legal note

Passive scanning is generally lawful, but laws on **recording, decoding and disclosing** certain transmissions (mobile, police/emergency, aviation, encrypted traffic) differ by country. Check your local rules, especially before keeping IQ captures of voice or data. Never transmit with this tool; it only receives.

## Credits

Signal database: [Artemis-DB](https://github.com/AresValley/Artemis), built from [SigID Wiki](https://www.sigidwiki.com). It is downloaded by the user and is not redistributed here.

## Tests

```bash
python3 -m unittest discover -s tests
```
