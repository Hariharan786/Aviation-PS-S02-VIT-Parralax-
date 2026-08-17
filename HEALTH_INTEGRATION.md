# Bearing-integrated health and physics-based flight timing

## What changed

1. Bearing vibration features are calculated for every engine/cycle from the existing C-MAPSS physics telemetry.
2. The bearing result is merged into the same per-cycle RUL/anomaly table used by AeroGuard.
3. Overall health uses:
   - 65% existing ensemble RUL health
   - 20% anomaly-health contribution
   - 15% bearing-health contribution
4. Effective RUL is reduced by up to 25% according to bearing risk.
5. Maintenance recommendations are recalculated from the integrated status, bearing risk, and effective RUL.
6. Every engine gets an engine-specific recommendation and can be exported from the dashboard.

## Variable flight time

C-MAPSS provides cycle numbers rather than aircraft wall-clock timestamps or route distance. The implementation therefore estimates a mission-segment duration from true airspeed derived from Mach and altitude, throttle/load proxy, altitude change between adjacent cycles, and a mission-distance proxy. `cycle_duration_s` is intentionally variable and is accumulated into `flight_time_s_cumulative` and `flight_time_min_cumulative`.

The high-frequency bearing waveform remains a short 0.5-second diagnostic capture per cycle so CSV exports remain practical. Its mission position is carried by the variable flight-time fields.

This is a physics-informed simulation layer and should not be treated as certified aircraft maintenance data.
