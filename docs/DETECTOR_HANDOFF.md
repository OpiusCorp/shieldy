# Detector improvement handoff

## Review package

- Classifier: https://github.com/OpiusCorp/shieldy/pull/5 (`24400ab`). Corrects the
  bundled tensor contract, CSV parsing, firearm class selection, interpreter
  serialization, and native buffer reuse. Model weights and thresholds are unchanged.
- Capture and timing: https://github.com/OpiusCorp/shieldy/pull/6 (`f2a67d2`). Adds
  native 16 kHz capture with fallback, continuous peer pre-roll, stateful resampling,
  a shared event scheduler, and bounded tail collection.
- Evaluation: https://github.com/OpiusCorp/shieldy/pull/7 (`af6564b`). Provides a
  seeded synthetic corpus, direct TFLite inference, replay/scoring, manifests for
  external recordings, dependencies, and raw simulation results.

The three reviewed diffs apply together in an isolated checkout. The bundled asset
contract check passes and all 11 evaluation Python modules pass syntax compilation.
The service author reports 14 passing unit tests and a successful debug APK build
at `f2a67d2`. These Android results were not independently reproduced by the lead
reviewer because their environment lacks Java. Python inference tests were not
independently reproduced there because the scientific/TensorFlow dependencies are absent.

## Demo and verification

After integrating the PRs into a review branch:

1. Run `python3 tools/check_detector_assets.py`.
2. Install `eval/requirements.txt` in a Python virtual environment. Run
   `python3 -m pytest eval/test_scoring.py eval/test_inference.py` and
   `python3 -m eval.run_eval --compare`.
3. Run `./gradlew :app:testDebugUnitTest :app:assembleDebug` with Java 17 and the
   Android SDK configured.
4. On real Android phones, verify microphone permissions, listening telemetry,
   peer discovery, and consensus behavior. Confirm that non-sentinel history remains
   available and an early peer request waits for the event tail. Use recorded audio
   or synthetic signals for demonstration; do not use live firearms.

## What the results establish

The persisted synthetic simulation reports six of six generated threat events
detected with tightened classes and tail waiting, with one false alert on a
46-second negative stress stream, compared with five in its simulated baseline.
Median simulated detection latency is about 322 ms. The polyphase experiment misses
one generated threat event and is not a recommended model change on this evidence.

The stored results identify an older detector SHA, not final Android revision
`f2a67d2`. They demonstrate the harness and a timing hypothesis, not measured Android
or field accuracy. The Python simulator is separate from the Kotlin implementation.
One false alert in 46 seconds is also far too frequent to establish deployment fitness.

## Remaining validation

Before treating this as a reliable detector, evaluate licensed labeled recordings
with source/session-separated splits, verify the final Android revision on devices,
and record model/corpus hashes, exact source SHAs, recall, false alerts per listening
hour, and end-to-end latency. Custom-head training and threshold tuning are deferred
until this baseline exists. This package is a prototype improvement, not a validated
emergency detection system.
