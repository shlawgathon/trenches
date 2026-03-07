# DATA.md: Trenches Entity Source Inventory

This file tracks the sources currently connected into `trenches` for each simulator entity.

Rules for this inventory:

- Every entity has at least 20 connected sources.
- The frontend registry in `src/lib/data-sources/registry.ts` is the source of truth.
- The backend RL environment mirrors the same names in `backend/src/trenches_env/source_bundles.py`.
- `training_core` sources are intended for episodic RL observations.
- `live_demo` sources are intended for live sessions, monitoring, and demo overlays.

World Monitor provenance for these assignments came from the local `worldmonitor` clone, especially:

- `scripts/rss-feeds-report.csv`
- `data/telegram-channels.json`
- `src/components/LiveNewsPanel.ts`
- `src/components/LiveWebcamsPanel.ts`
- `docs/DATA_SOURCES.md`
- generated `/api/.../v1/...` service routes

## US

Count: 20 total

Training core:

- Reuters US
- Politico
- Wall Street Journal
- White House
- State Dept
- USNI News
- Polymarket Geopolitical Markets
- NPR News
- Pentagon
- Treasury
- Federal Reserve
- SEC
- Defense One
- Defense News
- Military Times

Live/demo:

- Fox News Live
- Washington DC Webcam
- ABC News Live
- CBS News Live
- NBC News Live

## Israel

Count: 20 total

Training core:

- OREF Rocket Alerts
- Haaretz
- OpenSky Military Flights
- Wingbits Flight Enrichment
- TLV NOTAM / Airport Closures
- GPSJam Levant View
- The Defender Dome
- Yedioth News
- BBC Middle East
- The Times of Israel
- Levant Theater Posture
- Israel Frontier Base Layer
- Eastern Med Navigational Warnings

Live/demo:

- Kan 11 Live
- i24NEWS Live
- Jerusalem Webcam
- Tel Aviv Webcam
- Al Jazeera English Live
- Al Hadath Live
- Asharq News Live

## Iran

Count: 20 total

Training core:

- VahidOnline
- BBC Persian
- Iran International
- Fars News
- LiveUAMap Iran Events
- NASA FIRMS Strike Heat
- Fotros Resistance
- Iran International Telegram
- BNO News
- Al Arabiya
- Iran Nuclear Energy Watch
- Iran Internet Outages
- Iran Unrest Events
- Iran Climate Anomalies
- Iran Stock Index

Live/demo:

- Iran International Live
- Tehran Webcam
- TRT World Live
- CGTN Arabic Live
- France 24 Live

## Hezbollah

Count: 20 total

Training core:

- Abu Ali Express
- Abu Ali Express EN
- Lebanon Update
- Middle East Spectator
- The Cradle
- ACLED Lebanon/Syria Conflict Events
- Middle East Now Breaking
- Aurora Intel
- OSINTdefender
- OSIntOps News
- OSINT Live
- Guardian Middle East
- Al Jazeera Arabic
- Lebanon Humanitarian Summary
- Lebanon Unrest Events
- Lebanon Internet Outages
- Lebanon Climate Anomalies
- Lebanon Coast Navigational Warnings

Live/demo:

- Beirut MTV Lebanon Webcam
- Rudaw Live

## Gulf Coalition

Count: 20 total

Training core:

- Arabian Business
- The National (GCC Query Set)
- Gulf Investments
- Gulf Economies Panel
- Oil and Energy Analytics
- Maritime Chokepoint Disruption Panel
- Gulf FDI Layer
- Arab News
- Reuters Business
- CNBC
- Yahoo Finance
- Shipping Rates Monitor
- Critical Minerals Monitor
- Trade Restrictions Monitor
- Tariff Trends Monitor

Live/demo:

- Sky News Arabia Live
- Mecca Webcam
- Al Arabiya Live
- Al Jazeera Arabic Live
- Middle East Regional Webcam

## Oversight

Count: 20 total

Training core:

- Country Instability Index (CII)
- Hotspot Escalation Score
- Strategic Risk Score
- Cross-Stream Correlation Engine
- Intelligence Gap Tracker
- Headline Memory / World Brief / AI Deduction
- HAPI Displacement Data
- WorldPop Population Exposure
- Security Advisories Aggregation
- UCDP Conflict Events
- Natural Events Monitor
- Earthquakes Feed
- Internet Outages Baseline
- Cable Health Advisory Layer
- Climate Anomalies Monitor
- Cyber Threats Feed
- Feed Digest Aggregator
- GDELT Document Search
- Pizzint Status

Live/demo:

- UNHCR Feed
