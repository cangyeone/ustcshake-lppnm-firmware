# Changelog

## v2.0.0-alpha.1 — 2026-10-30

Initial experimental firmware release for the educational seismometer PCB V2.

- Added 3072/128 LPPNM fused W8A8 incremental inference and stateful 250→100 Hz resampling.
- Rebuilt sampling, filtering, range selection, three-button menus, USB streaming and time portal.
- Added horizontal display with retrospective P/S markers; corrected display rotation after device-user feedback.
- Set P/S thresholds to 0.01, with on-screen inference warmup/probability diagnostics.
- Added arrival-aligned paper raster markers, confidence and millisecond timestamps, using a six-second queue.
- Passed host tests and a two-minute board stress run including 7,500 simulated waveform rows and ten phase annotations.
- Fixed the audit reader to assemble complete newline-delimited USB frames across serial timeouts.

The paper preview is generated test data. Real-paper validation of the new annotation layout and longer-duration/labelled-data accuracy testing remain outstanding.
