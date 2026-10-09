# hackrf-scout

Sweep the whole HackRF range (1 MHz – 6 GHz), find signals, work out what they probably are, and keep everything in a SQLite database. Optionally record a short IQ capture of each new signal.

```
hackrf_sweep ──► detect (noise floor + SNR + persistence) ──► identify (Artemis/SigID Wiki + bandplan) ──► SQLite
                                                                                                    └──► IQ captures (SigMF)
```

## Install

```bash
sudo apt install hackrf          # provides hackrf_sweep and hackrf_transfer
pip install -e .                 # needs Python 3.9+ and numpy
pip install -e '.[web]'          # optional: adds the browser interface (FastAPI + uvicorn)
hackrf-scout update-db           # one-off: downloads the SigID Wiki database (~300 MB), keeps ~1 MB of JSON
```

`update-db` downloads the Artemis-DB release, verifies its SHA-256 against the published value, opens it read-only and writes `data/artemis_signals.json`. The tool works without it, using only the built-in bandplan (EU / ITU Region 1), but names will be much more generic.

## Use

```bash
# Scan everything, keep scanning until Ctrl-C. New signals are printed as they are confirmed.
hackrf-scout scan --region-keywords greece,cyprus

# Only some ranges, with the RF amp on (weak signals)
hackrf-scout scan -f 400:470 -f 860:960 -a

# Or by band name (see "Bands" below): 433 MHz ISM, 868 MHz SRD and the 2.4 GHz band
hackrf-scout scan -f 433 -f 868 -f 2.4ghz

# Scan 10 minutes, then record 5 s of IQ from the best new signals, repeat forever
hackrf-scout run --scan-seconds 600 --capture-seconds 5 --capture-max 5

# Look at what was found
hackrf-scout report                     # everything
hackrf-scout report --unidentified -v   # the interesting ones, with candidate matches
hackrf-scout report --band 430:440 --sort max_snr
hackrf-scout report --band 868          # a band name works anywhere a range does
hackrf-scout anomalies                  # signals that are louder, quieter, gone quiet or new compared with their own baseline
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

## Web interface

```bash
pip install -e '.[web]'
hackrf-scout web                      # http://127.0.0.1:8765, watch and browse only
hackrf-scout web --allow-control      # also start and stop the scanner from the browser
```

Run it on the machine that has the HackRF and `scout.db` (SQLite's WAL mode does not work over network drives). It can run next to a scan started from a terminal.

| Tab | What it does |
|---|---|
| **Live** | Scanner status, counters, and the **live log**: new signals, sweep progress, the output of `hackrf_sweep`, errors. With `--allow-control` it also has the start/stop form. |
| **Signals** | Every stored signal, filtered by band, identification, hits, SNR or text; sortable and paged. Click a row for candidates, bands, signal-strength history and observations. CSV/JSON export of the current filter. |
| **Bands** | One row per band (433, 868, 2.4 GHz, ...): signal count, unidentified count, best SNR, activity over time, and the `scan -f` command that revisits only that band. |
| **Data** | Read-only view of every table in the database (`signals`, `observations`, `sweeps`, `captures`, `scans`, `log`, `meta`). |

### Bands

Anywhere a range is accepted (`report --band`, `scan -f`, the web filters) you can use a band name or `START:STOP` in MHz. Presets include `433`, `868`, `2.4ghz`, `5.8ghz`, `pmr446`, `dect`, `fm`, `airband`, `marine`, `ham2m`, `ham70cm`, `tetra`, `gsm900`, `gsm1800`, `adsb`, `gnss`, and the umbrella bands `vhf` (30-300 MHz) and `uhf` (300-3000 MHz). UHF is almost half the spectrum, so the practical sub-bands are usually what you want. The full list is at `/api/bands` or in `hackrf_scout/bands.py`.

Add your own in `~/.hackrf-scout/bands.json` (or `--bands-file`, or `HACKRF_SCOUT_BANDS`):

```json
{"bands": [{"key": "garage", "name": "Garage remotes", "lo_mhz": 433.0, "hi_mhz": 435.0, "group": "Mine"}]}
```

The web filters match a signal if **any part of its bandwidth** overlaps the band, so a 20 MHz Wi-Fi channel centred just under 2400 MHz still counts for `2.4ghz`. Switch to "centre frequency inside band" if you prefer the strict reading (the CLI `report --band` always uses the centre).

When you scan, `hackrf_sweep` needs whole MHz, so a band is rounded outwards (433.05-434.79 MHz scans as `433:435`).

### Start and stop from the browser

* The server starts `hackrf-scout scan` or `run` as a **separate, detached process**, so a scan keeps going if you close the browser or restart the web server. Stop sends SIGINT: the scanner finishes its current sweep, commits and exits (SIGTERM, then SIGKILL, only if it does not).
* The HackRF can only be used by one program, so every `scan`, `run` and `capture` now holds a lock (`~/.hackrf-scout/scanner.lock`, a kernel `flock`, so a crash never leaves it stuck). A second start gets a clear error; the web UI shows scans started from a terminal too, and can stop them. Replaying a file with `scan --source` needs no hardware and takes no lock.
* Stopping loses signal candidates that have not yet reached `--min-hits`; confirmed signals and the sweep counter are kept.
* The form only offers receive modes and fixed, validated ranges. The database, executables and capture folder come from the server's command line (`--db`, `--hackrf-sweep`, `--hackrf-transfer`, `--capture-dir`), never from the browser. Linux and macOS only.

### Security

* Listens on `127.0.0.1` only. Another address needs `--token` (or `HACKRF_SCOUT_TOKEN`); the server refuses to start without one.
* `--allow-control` is off by default. When on and no token is given, a random one is generated and printed as a link (`http://127.0.0.1:8765/#token=...`); open that link once. The token travels in an `Authorization` header and is kept in the tab's session storage, not in cookies or URLs.
* The `Host` header must be local (DNS rebinding protection; add names with `--allowed-host`), cross-site POSTs are rejected, and a strict Content-Security-Policy applies (no inline scripts, no external resources).
* The database is opened read-only, table and column names are whitelisted, and there is no SQL console. IQ recordings are not served over HTTP: your legal note below applies to who can reach them.

### API

JSON under `/api` (all `GET` unless noted): `status`, `anomalies`, `signals` (`band`, `mode`, `unidentified`, `flagged`, `min_hits`, `min_snr`, `q`, `sort`, `order`, `page`, `page_size`), `signals/{id}`, `signals/export?format=csv|json`, `observations`, `sweeps`, `captures`, `bands`, `bands/summary`, `bands/activity`, `tables`, `tables/{name}`, `log?tail=N|after=ID`, `log/stream` (server-sent events, resumes with `Last-Event-ID`), `scanner`, and `POST scanner/start|stop|check` (only with `--allow-control`).

## Steadier detection

Three things keep a signal from flickering in and out, and keep a bad sweep from creating ghosts. All are on by default and have a switch:

| What | Default | How it works |
|---|---|---|
| **Smoothed noise floor** (`--floor-alpha`) | 0.2 | Each sweep's floor estimate wobbles by a dB or so, which moves signals across the SNR threshold. The floor is averaged over sweeps, per frequency (1 = off). |
| **Hysteresis** (`--hysteresis`) | 3 dB | A new signal must reach `--snr`, but a *confirmed* one stays detected down to `--snr` minus this, so a signal hovering at the threshold keeps its hit count. Weak detections can only continue a confirmed signal, never start one. |
| **Overload guard** (`--overload-db`) | 6 dB | `hackrf_sweep` cannot report clipping, so the guard looks for its symptoms: the whole noise floor jumping by more than this, or a flood of detections (over 30 and 4x the recent median). Such a sweep is skipped, with a warning in the log. If it lasts three sweeps in a row it is taken as the new normal (new antenna, new gain). 0 turns the guard off. |

## Baselines and alerts

Compared with *its own* history, is a signal behaving? Everything is computed from data already stored (`observations` and the new `scans` table), so it works while a scan runs, in the web UI and from the command line:

```bash
hackrf-scout anomalies [--kind louder] [--json]
```

| Flag | Meaning |
|---|---|
| `louder` / `quieter` | The median SNR of the last 5 observations is well away from the signal's baseline (the median of its older observations, with a robust spread). The shift must be at least 3 robust standard deviations and at least 3 dB, and the spread is never trusted below 1.5 dB (so in practice about 4.5 dB), which keeps small wobbles from counting. The median of five makes one blip harmless. Needs 13 observations first. |
| `gone_quiet` | A signal that normally appears at least every few minutes has not been seen for 5 times its longest usual gap (and at least 5 minutes) **of scanning that frequency**. |
| `new_in_quiet` | First seen in the last hour, where nothing else had ever been seen within 1 MHz (or its own bandwidth), and that frequency had been scanned for at least 30 minutes before that. |

* **"Scanned" matters.** Each scan records its frequency ranges (`scans` table). Scanning a different band does not make everything look silent or new, and stopping the scanner does not make everything "gone quiet": the judgements are made as of the last moment the scanner was looking. The age is shown next to the alerts.
* SNR (against the local noise floor) is compared, not raw power, so a gain change between sessions is not mistaken for a signal change.
* While a scan runs, each new flag is written to the log once a minute at most (`ALERT ...` / `notice ...`). In the web UI they show on the Live tab, as badges in the Signals table (with a "Flagged only" filter) and in each signal's detail.
* Thresholds are in `hackrf_scout/baseline.py` (`BaselineConfig`). They are reasonable starting points, not tuned on real captures; expect to adjust them for your site.

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
| A real signal keeps dropping out at the threshold | `--hysteresis 4` (or higher), or a slightly lower `--snr` |
| "sweep skipped, possible receiver overload" in the log | A strong transmitter is overloading the front end: lower `-l`/`-g`, turn `-a` off, move the antenna. If the new level is real, the guard accepts it after 3 sweeps; `--overload-db 0` turns it off |
| Frequency resolution | `-w 100000` is the default; smaller bins resolve narrow signals better but slow the sweep |

## Limits

* Power values are relative dB from `hackrf_sweep`, not calibrated dBm.
* `hackrf_sweep` and `hackrf_transfer` cannot use the HackRF at the same time, so IQ capture only happens between scan cycles (`run`). Signals that are not transmitting during the capture window produce silent recordings.
* The HackRF has 8-bit ADC and image/spur products. Very strong signals produce ghost copies; check anything suspicious with a lower gain.
* Only what is on the air during the scan is seen. Intermittent signals need long runs.
* The simulator tests the pipeline, not real-world false-positive rates. Expect to tune `--snr`, `--min-hits` and `--ignore` for your location.

## Files

* `scout.db`: SQLite with tables `signals`, `observations`, `sweeps`, `captures`, `scans` (one row per scan session with its frequency ranges, used by the baselines), `log` (the last 5000 log lines, shown in the web UI), `meta`.
* `~/.hackrf-scout/`: `scanner.lock` and `scanner.json` (who is using the HackRF), `scanner.out` (console output of a scan started from the browser), optional `bands.json`. Override the folder with `HACKRF_SCOUT_STATE_DIR`.
* `captures/`: IQ recordings in [SigMF](https://sigmf.org). Each is a pair: `sigNNNN_<freq>MHz_<time>.sigmf-data` is signed 8-bit interleaved I/Q, byte for byte what `hackrf_transfer` writes (SigMF calls it `ci8`), and the `.sigmf-meta` next to it holds the sample rate, centre frequency, UTC start time, gains, a SHA-512 of the data, and an annotation with the signal's estimated frequency edges and what it was identified as. SigMF-aware tools need nothing else; a tool that wants raw I/Q can read the `.sigmf-data` file as it is.

## Legal note

Passive scanning is generally lawful, but laws on **recording, decoding and disclosing** certain transmissions (mobile, police/emergency, aviation, encrypted traffic) differ by country. Check your local rules, especially before keeping IQ captures of voice or data. Never transmit with this tool; it only receives.

## Credits

Signal database: [Artemis-DB](https://github.com/AresValley/Artemis), built from [SigID Wiki](https://www.sigidwiki.com). It is downloaded by the user and is not redistributed here.

## Tests

```bash
python3 -m unittest discover -s tests
```

The web tests need `pip install -e '.[web,dev]'` and are skipped without it. They use fake `hackrf_*` programs, so no hardware is involved.
