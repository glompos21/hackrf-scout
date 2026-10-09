"""Built-in, *indicative* frequency allocations (ITU Region 1 / EU, HackRF range).

Used as a fallback when the Artemis/SigID database has no good match, and to
give every detection a human-readable "what lives here" hint. Allocations
differ by country; check your national frequency plan before relying on a label.
"""

from __future__ import annotations

from typing import List, Tuple

# (low MHz, high MHz, name)
_RAW: List[Tuple[float, float, str]] = [
    # --- HF / low VHF ---
    (1.0, 1.606, "MW broadcast (AM)"),
    (1.81, 2.0, "Amateur 160 m"),
    (3.5, 3.8, "Amateur 80 m"),
    (5.9, 6.2, "Shortwave broadcast 49 m"),
    (7.0, 7.2, "Amateur 40 m"),
    (9.4, 9.9, "Shortwave broadcast 31 m"),
    (10.1, 10.15, "Amateur 30 m"),
    (14.0, 14.35, "Amateur 20 m"),
    (18.068, 18.168, "Amateur 17 m"),
    (21.0, 21.45, "Amateur 15 m"),
    (24.89, 24.99, "Amateur 12 m"),
    (26.965, 27.405, "CB radio (27 MHz)"),
    (28.0, 29.7, "Amateur 10 m"),
    (30.0, 88.0, "VHF low band (land mobile / government)"),
    (50.0, 52.0, "Amateur 6 m"),
    (87.5, 108.0, "FM broadcast"),
    (108.0, 118.0, "Aeronautical navigation (VOR/ILS)"),
    (118.0, 137.0, "Airband voice (AM)"),
    (121.4, 121.6, "Aeronautical distress 121.5"),
    (137.0, 138.0, "Weather satellites (NOAA APT / Meteor LRPT)"),
    (138.0, 144.0, "VHF government / military"),
    (144.0, 146.0, "Amateur 2 m"),
    (156.0, 162.05, "Marine VHF"),
    (156.775, 156.825, "Marine distress ch.16"),
    (161.9, 162.1, "AIS (vessel tracking)"),
    (162.05, 174.0, "VHF land mobile / PMR"),
    (174.0, 230.0, "VHF band III (DAB+ / TV)"),
    (225.0, 400.0, "Military air UHF"),
    (380.0, 400.0, "TETRA (emergency services)"),
    (406.0, 406.1, "Emergency beacons (COSPAS-SARSAT)"),
    (430.0, 440.0, "Amateur 70 cm"),
    (433.05, 434.79, "ISM 433 MHz (SRD: sensors, remotes, TPMS)"),
    (446.0, 446.2, "PMR446 walkie-talkies"),
    (470.0, 694.0, "UHF TV (DVB-T/T2)"),
    (694.0, 790.0, "LTE 700 MHz (band 28)"),
    (791.0, 821.0, "LTE 800 downlink (band 20)"),
    (832.0, 862.0, "LTE 800 uplink (band 20)"),
    (863.0, 870.0, "SRD 868 MHz (LoRa/LoRaWAN, sensors)"),
    (876.0, 880.0, "GSM-R uplink (railway)"),
    (880.0, 915.0, "GSM900 / LTE 900 uplink"),
    (921.0, 925.0, "GSM-R downlink (railway)"),
    (925.0, 960.0, "GSM900 / LTE 900 downlink"),
    (960.0, 1215.0, "Aeronautical radionavigation (DME/TACAN/SSR)"),
    (1088.0, 1092.0, "ADS-B / Mode S (1090 MHz)"),
    (1164.0, 1215.0, "GNSS L5 / E5"),
    (1215.0, 1300.0, "GNSS L2 / E6 / radar"),
    (1240.0, 1300.0, "Amateur 23 cm"),
    (1525.0, 1559.0, "Satellite downlink (Inmarsat/MSS)"),
    (1559.0, 1610.0, "GNSS L1 / E1 / GLONASS"),
    (1574.0, 1577.0, "GPS L1"),
    (1616.0, 1626.5, "Iridium satellite"),
    (1626.5, 1660.5, "Satellite uplink (MSS)"),
    (1710.0, 1785.0, "GSM1800 / LTE 1800 uplink"),
    (1805.0, 1880.0, "GSM1800 / LTE 1800 downlink"),
    (1880.0, 1900.0, "DECT cordless phones"),
    (1920.0, 1980.0, "UMTS / LTE 2100 uplink"),
    (2110.0, 2170.0, "UMTS / LTE 2100 downlink"),
    (2400.0, 2483.5, "ISM 2.4 GHz (Wi-Fi, Bluetooth, drones, Zigbee)"),
    (2500.0, 2570.0, "LTE 2600 uplink"),
    (2620.0, 2690.0, "LTE 2600 downlink"),
    (2700.0, 3100.0, "S-band radar (ATC / weather)"),
    (3400.0, 3800.0, "5G NR n78 (3.5 GHz)"),
    (5150.0, 5350.0, "Wi-Fi 5 GHz (UNII-1/2)"),
    (5470.0, 5725.0, "Wi-Fi 5 GHz (UNII-2e) / weather radar"),
    (5725.0, 5875.0, "ISM 5.8 GHz (Wi-Fi, FPV video, drones)"),
]

BANDPLAN = [(lo * 1e6, hi * 1e6, name) for lo, hi, name in _RAW]


def lookup(center_hz: float, bw_hz: float = 0.0) -> List[str]:
    """Names of all allocations containing the signal centre, narrowest first."""
    hits = [(hi - lo, name) for lo, hi, name in BANDPLAN if lo - bw_hz / 2 <= center_hz <= hi + bw_hz / 2]
    hits.sort()
    return [name for _, name in hits]
