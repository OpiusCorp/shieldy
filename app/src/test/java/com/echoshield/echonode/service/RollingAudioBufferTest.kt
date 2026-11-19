package com.echoshield.echonode.service

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import kotlin.math.sin

class RollingAudioBufferTest {

    private lateinit var buffer: RollingAudioBuffer

    @Before
    fun setUp() {
        buffer = RollingAudioBuffer(capacity = 64000, targetSampleRate = 16000)
    }

    @Test
    fun testSequentialWriteAndLatestWindowExtraction() {
        val windowSize = 1000
        val input = FloatArray(windowSize) { it.toFloat() / 1000f }
        buffer.write(input, windowSize, 1000L)

        val extracted = buffer.snapshotLatestWindow(windowSize)
        assertNotNull(extracted)
        assertEquals(windowSize, extracted!!.size)
        for (i in 0 until windowSize) {
            assertEquals(input[i], extracted[i], 1e-6f)
        }
    }

    @Test
    fun testBufferWrapAroundMaintainsChronologicalOrder() {
        val capacity = 1000
        val smallBuffer = RollingAudioBuffer(capacity = capacity, targetSampleRate = 16000)
        val totalToWrite = 2500
        val chunk = FloatArray(100)

        var timeMs = 0L
        for (batch in 0 until totalToWrite / 100) {
            for (i in 0 until 100) {
                chunk[i] = (batch * 100 + i).toFloat()
            }
            timeMs += 10L
            smallBuffer.write(chunk, 100, timeMs)
        }

        val windowSize = 500
        val window = smallBuffer.snapshotLatestWindow(windowSize)
        assertNotNull(window)
        assertEquals(windowSize, window!!.size)

        val expectedStart = (totalToWrite - windowSize).toFloat()
        for (i in 0 until windowSize) {
            assertEquals(expectedStart + i, window[i], 1e-4f)
        }
    }

    @Test
    fun testSnapshotWindowAtValidHistoricalOffset() {
        val windowSize = 1000
        val totalSamples = 16000
        val input = ShortArray(totalSamples) { (it % 1000).toShort() }

        buffer.write(input.copyOfRange(0, 8000), 8000, 500L)
        buffer.write(input.copyOfRange(8000, 16000), 8000, 1000L)

        val extracted = buffer.snapshotWindowAt(requestedEndTimeMs = 600L, windowSamples = windowSize)
        assertNotNull("Historical window at 600ms should be available", extracted)
        assertEquals(windowSize, extracted!!.size)

        val expectedStartSample = (600 * 16) - windowSize
        for (i in 0 until windowSize) {
            val expectedVal = (input[expectedStartSample + i] / 32768f)
            assertEquals(expectedVal, extracted[i], 1e-5f)
        }
    }

    @Test
    fun testSnapshotWindowAtExpiredHistoryReturnsNull() {
        val chunk = ShortArray(16000) { 100 }
        for (sec in 1..5) {
            buffer.write(chunk, chunk.size, sec * 1000L)
        }

        val expired = buffer.snapshotWindowAt(requestedEndTimeMs = 500L, windowSamples = 1000)
        assertNull("Expired historical audio must return null and not clamp to oldest retained window", expired)
    }

    @Test
    fun testSnapshotWindowAtFutureTimestampReturnsNull() {
        val chunk = ShortArray(16000) { 100 }
        buffer.write(chunk, chunk.size, 1000L)

        val future = buffer.snapshotWindowAt(requestedEndTimeMs = 2000L, windowSamples = 1000)
        assertNull("Future timestamp must return null and not clamp to latest window", future)
    }

    @Test
    fun testSnapshotWindowBeforeSufficientSamplesReturnsNull() {
        val chunk = ShortArray(500) { 100 }
        buffer.write(chunk, chunk.size, 100L)

        val insufficient = buffer.snapshotWindowAt(requestedEndTimeMs = 100L, windowSamples = 1000)
        assertNull("Insufficient buffer samples must return null", insufficient)
    }

    @Test
    fun testDetectionSchedulerImpulseThenSilenceLatchesAndInferenceRunsWhenTailReady() {
        val scheduler = DetectionScheduler(
            windowTailMs = 300L,
            maxTailWaitMs = 450L,
            minInferenceIntervalMs = 400L
        )

        // t = 1000: Loud gunshot impulse opens gate
        scheduler.onFrame(currentTimeMs = 1000L, gateOpen = true, impulseOnsetMs = 1000L)
        assertTrue("Event must be latched as pending upon impulse", scheduler.isPending)
        assertEquals(1000L, scheduler.pendingFirstOnsetMs)
        assertFalse("Tail not ready yet at onset", scheduler.shouldRunInference(1000L, classifierReady = true))

        // t = 1050: Frame is silent (gate dropped to false)
        scheduler.onFrame(currentTimeMs = 1050L, gateOpen = false, impulseOnsetMs = null)
        assertTrue("Event must remain latched despite instantaneous gate closing", scheduler.isPending)
        assertFalse("Tail not ready at 1050ms", scheduler.shouldRunInference(1050L, classifierReady = true))

        // t = 1200: Still silent, tail not yet complete (300ms tail finishes at 1300ms)
        scheduler.onFrame(currentTimeMs = 1200L, gateOpen = false, impulseOnsetMs = null)
        assertFalse(scheduler.shouldRunInference(1200L, classifierReady = true))

        // t = 1300: 300ms reverberant tail has elapsed
        scheduler.onFrame(currentTimeMs = 1300L, gateOpen = false, impulseOnsetMs = null)
        assertTrue("Inference must run once 300ms tail is complete even with closed gate",
            scheduler.shouldRunInference(1300L, classifierReady = true))
        assertEquals(1300L, scheduler.getTargetWindowEndMs(1300L))

        // Complete inference
        scheduler.onInferenceCompleted(currentTimeMs = 1300L)
        assertFalse("Pending state must clear after inference completion", scheduler.isPending)

        // t = 1350: Subsequent silence does not trigger inference
        scheduler.onFrame(currentTimeMs = 1350L, gateOpen = false, impulseOnsetMs = null)
        assertFalse(scheduler.shouldRunInference(1350L, classifierReady = true))
    }

    @Test
    fun testDetectionSchedulerSustainedImpulsesTimeoutAnchoredToFirstOnset() {
        val scheduler = DetectionScheduler(
            windowTailMs = 300L,
            maxTailWaitMs = 450L,
            minInferenceIntervalMs = 400L
        )

        // Rapid series of impulse cracks: 1000, 1100, 1200, 1300, 1400
        scheduler.onFrame(currentTimeMs = 1000L, gateOpen = true, impulseOnsetMs = 1000L)
        assertEquals(1000L, scheduler.pendingFirstOnsetMs)

        scheduler.onFrame(currentTimeMs = 1100L, gateOpen = true, impulseOnsetMs = 1100L)
        scheduler.onFrame(currentTimeMs = 1200L, gateOpen = true, impulseOnsetMs = 1200L)
        scheduler.onFrame(currentTimeMs = 1300L, gateOpen = true, impulseOnsetMs = 1300L)
        scheduler.onFrame(currentTimeMs = 1400L, gateOpen = true, impulseOnsetMs = 1400L)

        // First onset remains anchored to 1000L, while latest onset is 1400L
        assertEquals(1000L, scheduler.pendingFirstOnsetMs)
        assertEquals(1400L, scheduler.pendingLatestOnsetMs)

        // At 1440ms, timeout (1000 + 450 = 1450) is not yet reached, and latest tail (1400 + 300 = 1700) is not reached
        assertFalse("At 1440ms, timeout has not elapsed", scheduler.isTailReady(1440L))

        // At 1450ms, timeout anchored to first onset expires (1450 - 1000 = 450 >= 450)
        assertTrue("At 1450ms, timeout anchored to first onset must fire despite sustained impulses",
            scheduler.isTailReady(1450L))
        assertTrue("Inference must proceed at timeout", scheduler.shouldRunInference(1450L, classifierReady = true))
    }

    @Test
    fun testDetectionSchedulerStaleImpulseDoesNotContaminateFutureRMSActivity() {
        val scheduler = DetectionScheduler(
            windowTailMs = 300L,
            maxTailWaitMs = 450L,
            minInferenceIntervalMs = 400L
        )

        // First event at 1000ms
        scheduler.onFrame(currentTimeMs = 1000L, gateOpen = true, impulseOnsetMs = 1000L)
        scheduler.onInferenceCompleted(currentTimeMs = 1300L)
        assertFalse(scheduler.isPending)

        // Long quiet period passes, then at 3000ms a new RMS-only sound occurs (no impulse onset)
        scheduler.onFrame(currentTimeMs = 3000L, gateOpen = true, impulseOnsetMs = null)
        assertTrue(scheduler.isPending)
        assertEquals("New event must anchor to current time, not old historical impulse",
            3000L, scheduler.pendingFirstOnsetMs)
        assertEquals(3000L, scheduler.pendingLatestOnsetMs)
    }

    @Test
    fun testDetectionSchedulerExpiredPendingEventResets() {
        val scheduler = DetectionScheduler(
            windowTailMs = 300L,
            maxTailWaitMs = 450L,
            minInferenceIntervalMs = 400L,
            eventLifecycleMs = 1500L
        )

        // Impulse at 1000ms, but classifier is never ready
        scheduler.onFrame(currentTimeMs = 1000L, gateOpen = true, impulseOnsetMs = 1000L)
        assertTrue(scheduler.isPending)

        // At 2600ms (1600ms after onset > 1500ms lifecycle), the unserviced event must expire
        scheduler.onFrame(currentTimeMs = 2600L, gateOpen = false, impulseOnsetMs = null)
        assertFalse("Pending event must expire after lifecycle window", scheduler.isPending)
    }

    @Test
    fun testPeerWakeRequestAtVariousOffsetsAfterOnset() {
        val windowSize = 1000
        val impulseAtMs = 1000L
        val windowTailMs = 300L
        val targetWindowEnd = impulseAtMs + windowTailMs // 1300L

        // Write audio incrementally into buffer to simulate continuous microphone capture
        // Buffer initially has audio up to 1000ms
        val initialChunk = ShortArray(16000) { 100 } // 1 second of audio (0..1000ms)
        buffer.write(initialChunk, initialChunk.size, 1000L)

        // Case 1: Peer request arrives at 0ms after onset (at 1000ms).
        // Target window 1300ms is in the future. Immediate snapshot returns null.
        val reqAt0ms = buffer.snapshotWindowAt(targetWindowEnd, windowSize)
        assertNull("At 0ms after onset, tail has not been recorded yet so snapshotWindowAt returns null", reqAt0ms)

        // Microphone records another 100ms (buffer at 1100ms)
        val chunk100 = ShortArray(1600) { 200 }
        buffer.write(chunk100, chunk100.size, 1100L)

        // Case 2: Peer request arrives at 100ms after onset (at 1100ms).
        // Target window 1300ms is still in the future.
        val reqAt100ms = buffer.snapshotWindowAt(targetWindowEnd, windowSize)
        assertNull("At 100ms after onset, target 1300ms is still future", reqAt100ms)

        // Microphone records another 100ms (buffer at 1200ms)
        val chunk200 = ShortArray(1600) { 300 }
        buffer.write(chunk200, chunk200.size, 1200L)

        // Case 3: Peer request arrives at 200ms after onset (at 1200ms).
        val reqAt200ms = buffer.snapshotWindowAt(targetWindowEnd, windowSize)
        assertNull("At 200ms after onset, target 1300ms is still future", reqAt200ms)

        // Microphone records another 150ms (buffer now at 1350ms, past 1300ms tail)
        val chunkTail = ShortArray(2400) { 400 }
        buffer.write(chunkTail, chunkTail.size, 1350L)

        // Case 4: Audio tail has arrived! Snapshot at target 1300ms now succeeds!
        val reqResolved = buffer.snapshotWindowAt(targetWindowEnd, windowSize)
        assertNotNull("Once buffer captures past 1300ms tail, snapshotWindowAt succeeds", reqResolved)
        assertEquals(windowSize, reqResolved!!.size)

        // Case 5: Peer request arrives at 600ms after onset (at 1600ms).
        // Microphone records up to 1600ms.
        val chunk600 = ShortArray(4000) { 500 }
        buffer.write(chunk600, chunk600.size, 1600L)
        val reqAt600ms = buffer.snapshotWindowAt(targetWindowEnd, windowSize)
        assertNotNull("At 600ms after onset, window 1300ms is readily available in retained history", reqAt600ms)
        assertEquals(windowSize, reqAt600ms!!.size)
    }

    @Test
    fun testResamplerChunkPartitionInvariance() {
        val resamplerFull = RollingAudioBuffer.Resampler(targetRate = 16000)
        val resamplerChunked = RollingAudioBuffer.Resampler(targetRate = 16000)

        val totalInput = 10000
        val input = ShortArray(totalInput) { i ->
            (sin(i * 0.05) * 30000).toInt().toShort()
        }

        val outputFull = resamplerFull.resampleToArray(input, totalInput, 44100)

        val outputChunkedList = mutableListOf<Float>()
        var offset = 0
        val chunkSizes = intArrayOf(64, 127, 255, 513, 1024, 73, 300, 100)
        var sizeIndex = 0

        while (offset < totalInput) {
            val chunkSize = minOf(chunkSizes[sizeIndex % chunkSizes.size], totalInput - offset)
            val chunk = input.copyOfRange(offset, offset + chunkSize)
            resamplerChunked.resampleAndFeed(chunk, chunkSize, 44100) { sample ->
                outputChunkedList.add(sample)
            }
            offset += chunkSize
            sizeIndex++
        }

        assertEquals(outputFull.size, outputChunkedList.size)
        for (i in outputFull.indices) {
            assertEquals("Sample $i differs between chunk partitions", outputFull[i], outputChunkedList[i], 1e-5f)
        }
    }

    @Test
    fun testResamplerLongStreamExactSampleCount() {
        val targetRate = 16000

        val resampler44k = RollingAudioBuffer.Resampler(targetRate)
        val in44kRate = 44100
        val in44kSamples = in44kRate * 10
        val chunk44k = ShortArray(1024) { (it % 1000).toShort() }

        var outputCount44k = 0
        var remaining44k = in44kSamples
        while (remaining44k > 0) {
            val toFeed = minOf(remaining44k, chunk44k.size)
            resampler44k.resampleAndFeed(chunk44k, toFeed, in44kRate) {
                outputCount44k++
            }
            remaining44k -= toFeed
        }
        assertEquals("44.1 kHz 10s stream must produce exactly 160,000 samples", 160000, outputCount44k)

        val resampler48k = RollingAudioBuffer.Resampler(targetRate)
        val in48kRate = 48000
        val in48kSamples = in48kRate * 10
        val chunk48k = ShortArray(1024) { (it % 1000).toShort() }

        var outputCount48k = 0
        var remaining48k = in48kSamples
        while (remaining48k > 0) {
            val toFeed = minOf(remaining48k, chunk48k.size)
            resampler48k.resampleAndFeed(chunk48k, toFeed, in48kRate) {
                outputCount48k++
            }
            remaining48k -= toFeed
        }
        assertEquals("48 kHz 10s stream must produce exactly 160,000 samples", 160000, outputCount48k)
    }

    @Test
    fun testResamplerFractionalPhasePreservedAcrossReads() {
        val resampler = RollingAudioBuffer.Resampler(targetRate = 16000)
        val sourceRate = 44100
        val chunk1 = ShortArray(2) { 16384 }
        var emittedCount = 0
        resampler.resampleAndFeed(chunk1, 2, sourceRate) {
            emittedCount++
        }
        assertEquals(0, emittedCount)

        val chunk2 = ShortArray(1) { 16384 }
        resampler.resampleAndFeed(chunk2, 1, sourceRate) {
            emittedCount++
        }
        assertEquals(1, emittedCount)
    }
}
