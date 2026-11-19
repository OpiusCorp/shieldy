# Gunshot detector validation

The Android detector still uses the bundled pretrained YAMNet weights. These changes
correct inference and class selection; they do not establish improved measured accuracy.

Only Gunshot/gunfire, Machine gun, Fusillade, and Artillery fire contribute to
gunshot confidence. Explosion, Bang, Cap gun, and other generic impulses remain
visible as top labels but cannot independently count as firearm evidence. CSV names
containing commas are preserved. Tensor input size/type and score output shape are
checked before inference, and inference/close are serialized across service callers.

## Evaluation before threshold tuning or replacing weights

1. Assemble labeled, licensed recordings of gunshots and hard negatives: doors,
   dropped objects, claps, balloons, fireworks, speech, music, and construction.
   Include quiet/distant shots, indoor reverberation, and clipped phone recordings.
2. Split by recording session, location, and source recording before making windows.
   Keep related clips and augmented copies in the same split to prevent leakage.
3. Replay continuous audio through the capture gate and rolling windows as well as
   testing isolated model windows. Include impulses at all positions relative to
   the 400 ms inference interval and peer classification delays.
4. Report event recall, false alerts per listening hour, precision, detection latency,
   inference failures, and results per phone/environment. Compare against the same
   baseline recordings; include confidence intervals and the number of events/hours.
5. Select thresholds on validation data only. Freeze them before measuring held-out
   test data. Evaluate clipped recordings separately because the service currently
   lowers its threshold for severe clipping.
6. For a custom model, train a small classifier on YAMNet embeddings using the training
   split, including the hard negatives above. Export a versioned asset with its class
   map, input/output contract, dataset provenance, and held-out evaluation report.

Run `./gradlew :app:testDebugUnitTest :app:compileDebugKotlin` for class-policy
regressions and compilation. Real-device inference, continuous-audio evaluation,
and peer-vote replay remain required before claiming detection improvements.

Run `python3 tools/check_detector_assets.py` without Android or TensorFlow installed
to verify the bundled model input/output contract and label-map indices. This check
does not execute inference or measure accuracy.
