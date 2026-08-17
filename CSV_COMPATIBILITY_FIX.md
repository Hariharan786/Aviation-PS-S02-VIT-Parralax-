
# CSV compatibility fix

AeroGuard now accepts all of these telemetry formats:

1. NASA C-MAPSS whitespace TXT
2. CSV with the 26-column C-MAPSS header
3. Headerless CSV in the exact 26-column C-MAPSS order
4. Tab-delimited telemetry

The frontend can therefore upload either:

- `FD001_physics_live_test.txt`
- `FD001_physics_live_test.csv`

The loader automatically detects the delimiter and header.

The important schema remains:

`engine_id, cycle, setting_1, setting_2, setting_3, T2, T24, T30, T50, P2, P15, P30, Nf, Nc, epr, Ps30, phi, NRf, NRc, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32`

The CSV header is consumed as a header, so the first data row is no longer
mistaken for `engine_id`. Both the LSTM prediction path and the anomaly
detection path use the same format-aware loader.
