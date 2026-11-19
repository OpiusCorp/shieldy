package com.echoshield.echonode.service

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import kotlin.math.abs
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

        // The last 500 samples written should be 2000..2499
        val expectedStart = (totalToWrite - windowSize).toFloat()
        for (i in 0 until windowSize) {
            assertEquals(expectedStart + i, window[i], 1e-4f)
        }
    }

    @Test
    fun testSnapshotWindowAtValidHistoricalOffset() {
        val windowSize = 1000
        val totalSamples = 16000 // 1 second of audio
        val input = ShortArray(totalSamples) { (it % 1000).toShort() }

        // Write in two 500ms batches
        buffer.write(input.copyOfRange(0, 8000), 8000, 500L)
        buffer.write(input.copyOfRange(8000, 16000), 8000, 1000L)

        // Request window ending at 600ms (400ms behind latest write)
        val extracted = buffer.snapshotWindowAt(requestedEndTimeMs = 600L, windowSamples = windowSize)
        assertNotNull("Historical window at 600ms should be available", extracted)
        assertEquals(windowSize, extracted!!.size)

        // At 600ms, samples ending at index (600 * 16) = 9600.
        // Window from 8600 to 9599.
        val expectedStartSample = (600 * 16) - windowSize
        for (i in 0 until windowSize) {
            val expectedVal = (input[expectedStartSample + i] / 32768f)
            assertEquals(expectedVal, extracted[i], 1e-5f)
        }
    }

    @Test
    fun testSnapshotWindowAtExpiredHistoryReturnsNull() {
        // Buffer capacity = 64000 samples = 4000ms at 16 kHz
        val chunk = ShortArray(16000) { 100 }
        // Write 5 seconds of audio (5 batches of 1 second each)
        for (sec in 1..5) {
            buffer.write(chunk, chunk.size, sec * 1000L)
        }

        // Requested window from 500ms (which is 4500ms behind lastWriteEndTimeMs of 5000ms).
        // Since capacity is 4000ms, history from 500ms has been overwritten.
        val expired = buffer.snapshotWindowAt(requestedEndTimeMs = 500L, windowSamples = 1000)
        assertNull("Expired historical audio must return null and not clamp to oldest retained window", expired)
    }

    @Test
    fun testSnapshotWindowAtFutureTimestampReturnsNull() {
        val chunk = ShortArray(16000) { 100 }
        buffer.write(chunk, chunk.size, 1000L)

        // Requested window at 2000ms (1000ms in the future relative to 1000L)
        val future = buffer.snapshotWindowAt(requestedEndTimeMs = 2000L, windowSamples = 1000)
        assertNull("Future timestamp must return null and not clamp to latest window", future)
    }

    @Test
    fun testSnapshotWindowBeforeSufficientSamplesReturnsNull() {
        val chunk = ShortArray(500) { 100 }
        buffer.write(chunk, chunk.size, 100L)

        // Requested window of 1000 samples when only 500 samples exist
        val insufficient = buffer.snapshotWindowAt(requestedEndTimeMs = 100L, windowSamples = 1000)
        assertNull("Insufficient buffer samples must return null", insufficient)
    }

    @Test
    fun testBoundedTailWaitDoesNotIndefinitelyPostponeInference() {
        val maxTailWaitMs = 450L
        val windowTailMs = 300L
        val firstImpulseAtMs = 1000L

        // Scenario: A sustained loud event repeatedly updates impulse onset
        var currentImpulseOnsetMs = firstImpulseAtMs
        var currentTimeMs = firstImpulseAtMs + 50L

        // Before tail or timeout: wait
        var tailTargetTimeMs = currentImpulseOnsetMs + windowTailMs
        var tailReady = (currentTimeMs >= tailTargetTimeMs) ||
            (currentTimeMs - firstImpulseAtMs >= maxTailWaitMs)
        assertEquals(false, tailReady)

        // Rapid subsequent impulses push currentImpulseOnsetMs forward
        currentTimeMs = firstImpulseAtMs + 400L
        currentImpulseOnsetMs = currentTimeMs // another loud crack
        tailTargetTimeMs = currentImpulseOnsetMs + windowTailMs
        tailReady = (currentTimeMs >= tailTargetTimeMs) ||
            (currentTimeMs - firstImpulseAtMs >= maxTailWaitMs)
        assertEquals(false, tailReady)

        // Once currentTime passes firstImpulseAtMs + maxTailWaitMs (450ms)
        currentTimeMs = firstImpulseAtMs + 460L
        tailReady = (currentTimeMs >= tailTargetTimeMs) ||
            (currentTimeMs - firstImpulseAtMs >= maxTailWaitMs)
        assertTrue("Inference must proceed once maxTailWaitMs timeout elapses despite sustained impulses", tailReady)
    }

    @Test
    fun testResamplerChunkPartitionInvariance() {
        val resamplerFull = RollingAudioBuffer.Resampler(targetRate = 16000)
        val resamplerChunked = RollingAudioBuffer.Resampler(targetRate = 16000)

        // Generate 10000 samples of 44.1 kHz sine wave
        val totalInput = 10000
        val input = ShortArray(totalInput) { i ->
            (sin(i * 0.05) * 30000).toInt().toShort()
        }

        // Process all at once
        val outputFull = resamplerFull.resampleToArray(input, totalInput, 44100)

        // Process in varied chunk sizes: 64, 127, 255, 512, etc.
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

        // Output sizes must match
        assertEquals(outputFull.size, outputChunkedList.size)

        // Output samples must match within small float epsilon
        for (i in outputFull.indices) {
            assertEquals("Sample $i differs between chunk partitions", outputFull[i], outputChunkedList[i], 1e-5f)
        }
    }

    @Test
    fun testResamplerLongStreamExactSampleCount() {
        val targetRate = 16000

        // Test 1: 10 seconds of 44.1 kHz (441,000 samples) -> exactly 160,000 samples
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

        // Test 2: 10 seconds of 48 kHz (480,000 samples) -> exactly 160,000 samples
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
        // Ratio = 44100 / 16000 = 2.75625
        // If we feed 2 samples, output sample count is 0 because 2.0 < 2.75625
        val chunk1 = ShortArray(2) { 16384 }
        var emittedCount = 0
        resampler.resampleAndFeed(chunk1, 2, sourceRate) {
            emittedCount++
        }
        assertEquals(0, emittedCount)

        // Feed 1 more sample (total 3 samples = 3.0 >= 2.75625)
        // This must emit exactly 1 sample and carry over 3.0 - 2.75625 = 0.24375 to the next read
        val chunk2 = ShortArray(1) { 16384 }
        resampler.resampleAndFeed(chunk2, 1, sourceRate) {
            emittedCount++
        }
        assertEquals(1, emittedCount)
    }
}
