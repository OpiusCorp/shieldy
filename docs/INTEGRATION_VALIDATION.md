# Combined application validation

## Outcome

PRs #5, #6, and #7 integrate cleanly. The combined Android app builds, passes its
automated checks, runs the real packaged model on an Android 14/API 34 emulator,
and launches its microphone foreground service. It is suitable for a controlled
physical-device pilot after review. These checks do not establish field gunshot
detection accuracy or authorize an operational emergency-system rollout.

Source inputs: #5 `acae376`, #6 `f2a67d2`, #7 `642c714`. The integration branch is
`test/detector-integration`. Test summaries and logs are in `docs/validation/`.

## Verified by the integration lead

- Android unit tests: **16 passed**, including the shared scheduler, rational
  resampler, historical buffer, and firearm label selection.
- Android debug and release builds: **passed**. Release output is unsigned;
  production signing is not configured by this work.
- Debug and release lint: **passed with warnings**, after fixing the explicit
  microphone permission check at AudioRecord initialization. Existing Compose
  lint API incompatibility skips some custom Compose checks; other warnings
  include unused resources, obsolete version guards, and dependency updates.
- Android instrumentation: **2 passed** on an arm64 Google APIs API 34 emulator.
  These execute the actual bundled TFLite asset, repeat inference, reinitialize
  the interpreter, and verify a known 440 Hz waveform after buffer reuse.
- Python scoring, actual TFLite inference, and provenance tests: **16 passed** after the final corrections.
- Asset contract: **passed**, input `[15600]`, output `[1, 521]`, 521 labels.
- Relay syntax check and full test suite: **32 passed**, including authentication,
  report privacy, peer quorum/voting, relay propagation, and triangulation.
- Compatible npm lockfile security updates applied; npm audit reports **0 known
  vulnerabilities** after updating vulnerable transitive dependencies.

Emulator UI/runtime smoke checks also passed: cold launch, Monitoring Active
dashboard, native 16 kHz AudioRecord initialization, YAMNet initialization with
four firearm classes, Test Alert transition to BARRICADE, foreground service
survival across Home/resume, and a denied-microphone-permission launch returning
to the permission screen. No AndroidRuntime crash was logged during these checks.
Test Alert is a manual debug path; it does not validate acoustic peer consensus.

## Reproduce

Configure Java 17 and Android SDK platform 34/build-tools 34.0.0, then run:

```sh
./gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
./gradlew :app:assembleRelease :app:lintRelease
# Requires an attached device or booted emulator:
./gradlew :app:connectedDebugAndroidTest
python3 tools/check_detector_assets.py
python3 -m venv .venv
.venv/bin/python -m pip install -r eval/requirements.txt
.venv/bin/python -m pytest eval/test_scoring.py eval/test_inference.py eval/test_provenance.py -q
.venv/bin/python -m eval.run_eval --compare --output /tmp/replay.json
cd server
npm ci
npm run check
npm test
npm audit
```

The lead used OpenJDK 17, Gradle 8.5, Node 24.20.0, Python 3.13, and TensorFlow
2.21.0. Dependencies in the evaluation requirements have lower bounds rather
than a fully pinned lock; preserve exact environment versions with future results.

## Replay result and limits

The fresh identical-corpus replay reproduced six of six generated threats with
tightened classes and tail waiting, one false alert in 46 seconds of negative
audio, and about 356 ms median simulated latency. Polyphase resampling missed
one of six threats. These tiny synthetic streams do not establish field accuracy.

The integration branch corrects the simulator timeout from 600 ms to Kotlin's
450 ms and replaces its hardcoded result SHA with actual Git source metadata,
working-tree status, source-file hashes, and the model hash. Replay output explicitly
identifies itself as Python simulation with real TFLite inference. It does not execute
the Kotlin service; full parity of capture, resampling, scheduling, and peer behavior
still requires device validation. The earlier replay log is retained as historical
evidence, with corrected replay JSON recorded separately.

## Remaining deployment gates

1. Test actual microphone capture and acoustic peer consensus on at least two
   physical phones, including 16 kHz initialization failure/fallback, interrupted
   capture, permission revocation, delayed peer requests, and app backgrounding.
2. Measure battery use and capture continuity under long listening sessions,
   screen lock, Doze, and supported Android versions. Short emulator checks do
   not establish sustained reliability.
3. Verify remaining simulator/device parity, then evaluate licensed labeled real
   recordings with disjoint recording/session splits and adequate negative hours.
4. Supply production relay HTTPS/authentication and Maps configuration, verify
   actual Android-to-deployed-relay connectivity, configure release signing, and
   review deployment-specific permissions/privacy. Default local builds have no
   production relay URL or Maps key.

No PR has been merged and no production deployment was performed by this pass.
