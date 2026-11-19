package com.echoshield.echonode.service

/**
 * Thread-safe circular audio buffer for the gunshot detector.
 *
 * Retains [capacity] samples at [targetSampleRate] (default 16 kHz).
 * Supports:
 * - Direct writing of normalized float samples (native 16 kHz capture).
 * - Direct writing of 16-bit PCM short samples (native 16 kHz capture).
 * - Continuous area-integrating resampling for fallback sample rates (e.g. 44.1 kHz, 48 kHz)
 *   preserving fractional phase across buffer reads and chunk boundaries to guarantee
 *   chunk-partition invariance and exact long-stream sample counts.
 * - Historical window extraction addressing wall-clock timestamps ([snapshotWindowAt]),
 *   which returns null for unavailable (expired or future) windows to prevent classifying
 *   unrelated audio.
 */
class RollingAudioBuffer(
    val capacity: Int = DEFAULT_CAPACITY,
    val targetSampleRate: Int = DEFAULT_SAMPLE_RATE
) {
    companion object {
        const val DEFAULT_SAMPLE_RATE = 16000
        const val DEFAULT_CAPACITY = 64000 // 4 seconds at 16 kHz
    }

    private val lock = Any()
    private val buffer = FloatArray(capacity)
    private var writeIndex = 0

    @Volatile
    var totalSamplesWritten: Long = 0L
        private set

    @Volatile
    var lastWriteEndTimeMs: Long = 0L
        private set

    val resampler = Resampler(targetSampleRate)

    /**
     * Resets the buffer and resampler state.
     */
    fun reset() {
        synchronized(lock) {
            writeIndex = 0
            totalSamplesWritten = 0L
            lastWriteEndTimeMs = 0L
            buffer.fill(0f)
            resampler.reset()
        }
    }

    /**
     * Returns true if at least [requiredSamples] have been written into the buffer.
     */
    fun hasEnoughSamples(requiredSamples: Int): Boolean {
        return totalSamplesWritten >= requiredSamples
    }

    /**
     * Writes [count] normalized float samples into the circular buffer.
     * Stamped with [frameEndTimeMs] corresponding to the end of the written batch.
     */
    fun write(samples: FloatArray, count: Int, frameEndTimeMs: Long) {
        if (count <= 0) return
        synchronized(lock) {
            val toWrite = minOf(count, samples.size)
            for (i in 0 until toWrite) {
                buffer[writeIndex] = samples[i]
                writeIndex = (writeIndex + 1) % capacity
            }
            totalSamplesWritten += toWrite
            lastWriteEndTimeMs = frameEndTimeMs
        }
    }

    /**
     * Writes [count] 16-bit PCM short samples into the circular buffer, normalizing to [-1.0f, 1.0f].
     */
    fun write(samples: ShortArray, count: Int, frameEndTimeMs: Long) {
        if (count <= 0) return
        synchronized(lock) {
            val toWrite = minOf(count, samples.size)
            for (i in 0 until toWrite) {
                buffer[writeIndex] = samples[i] / 32768f
                writeIndex = (writeIndex + 1) % capacity
            }
            totalSamplesWritten += toWrite
            lastWriteEndTimeMs = frameEndTimeMs
        }
    }

    /**
     * Resamples [count] 16-bit PCM short samples from [sourceSampleRate] to [targetSampleRate]
     * using continuous area-averaging (box filter anti-aliasing) and appends them to the buffer.
     */
    fun writeResampled(source: ShortArray, count: Int, sourceSampleRate: Int, frameEndTimeMs: Long) {
        if (count <= 0) return
        if (sourceSampleRate == targetSampleRate) {
            write(source, count, frameEndTimeMs)
            return
        }

        synchronized(lock) {
            resampler.resampleAndFeed(source, count, sourceSampleRate) { outSample ->
                buffer[writeIndex] = outSample
                writeIndex = (writeIndex + 1) % capacity
                totalSamplesWritten++
            }
            lastWriteEndTimeMs = frameEndTimeMs
        }
    }

    /**
     * Extracts a window of [windowSamples] ending at [requestedEndTimeMs] (wall-clock time).
     *
     * Returns null if:
     * - Fewer than [windowSamples] have been recorded.
     * - [requestedEndTimeMs] is in the future (> [lastWriteEndTimeMs]).
     * - [requestedEndTimeMs] is expired (older than available retained history).
     *
     * Returns a freshly allocated [FloatArray] with chronological samples when available.
     */
    fun snapshotWindowAt(requestedEndTimeMs: Long, windowSamples: Int): FloatArray? {
        synchronized(lock) {
            if (totalSamplesWritten < windowSamples || lastWriteEndTimeMs == 0L) {
                return null
            }

            // Future timestamps are unavailable (do not clamp to latest window)
            if (requestedEndTimeMs > lastWriteEndTimeMs) {
                return null
            }

            val msBehind = lastWriteEndTimeMs - requestedEndTimeMs
            val samplesBehind = (msBehind * targetSampleRate) / 1000L

            if (samplesBehind < 0L) {
                return null
            }

            val availableSamples = minOf(totalSamplesWritten, capacity.toLong())
            // Expired timestamps are unavailable (do not clamp to oldest window)
            if (samplesBehind + windowSamples > availableSamples) {
                return null
            }

            val end = Math.floorMod(writeIndex - samplesBehind.toInt(), capacity)
            val start = Math.floorMod(end - windowSamples, capacity)

            val output = FloatArray(windowSamples)
            val firstChunk = minOf(windowSamples, capacity - start)
            System.arraycopy(buffer, start, output, 0, firstChunk)
            if (firstChunk < windowSamples) {
                System.arraycopy(buffer, 0, output, firstChunk, windowSamples - firstChunk)
            }
            return output
        }
    }

    /**
     * Extracts the most recent window of [windowSamples].
     * Returns null if fewer than [windowSamples] have been recorded.
     */
    fun snapshotLatestWindow(windowSamples: Int): FloatArray? {
        return snapshotWindowAt(lastWriteEndTimeMs, windowSamples)
    }

    /**
     * Stateful area-averaging resampler.
     *
     * For downsampling (e.g. 44.1 kHz -> 16 kHz or 48 kHz -> 16 kHz):
     * Uses exact rational integer arithmetic (N = sourceRate / gcd, D = targetRate / gcd)
     * for input phase tracking, eliminating all floating-point rounding drift.
     *
     * Fractional state ([accumulatedInputWeight] and [currentOutputAcc]) is retained across
     * chunk calls, guaranteeing:
     * 1. Chunk-partition invariance: partitioning any stream into arbitrary chunk boundaries
     *    produces the exact same output samples.
     * 2. Zero sample-count drift: the number of output samples over long streams is exactly
     *    floor((totalInputSamples * targetRate) / sourceRate).
     */
    class Resampler(val targetRate: Int = DEFAULT_SAMPLE_RATE) {
        private var currentOutputAcc: Double = 0.0
        private var accumulatedInputWeight: Int = 0

        private fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

        fun reset() {
            currentOutputAcc = 0.0
            accumulatedInputWeight = 0
        }

        fun resampleAndFeed(
            source: ShortArray,
            count: Int,
            sourceRate: Int,
            consumer: (Float) -> Unit
        ) {
            val g = gcd(sourceRate, targetRate)
            val n = sourceRate / g // units needed per output sample
            val d = targetRate / g // units provided per input sample
            val toProcess = minOf(count, source.size)

            for (i in 0 until toProcess) {
                val v = source[i] / 32768.0
                var sampleRemaining = d

                while (sampleRemaining > 0) {
                    val needed = n - accumulatedInputWeight
                    if (sampleRemaining >= needed) {
                        currentOutputAcc += needed.toDouble() * v
                        sampleRemaining -= needed
                        val out = (currentOutputAcc / n.toDouble()).toFloat().coerceIn(-1.0f, 1.0f)
                        consumer(out)
                        currentOutputAcc = 0.0
                        accumulatedInputWeight = 0
                    } else {
                        currentOutputAcc += sampleRemaining.toDouble() * v
                        accumulatedInputWeight += sampleRemaining
                        sampleRemaining = 0
                    }
                }
            }
        }

        fun resampleToArray(source: ShortArray, count: Int, sourceRate: Int): FloatArray {
            val estimatedSize = ((count.toLong() * targetRate) / sourceRate).toInt() + 4
            val temp = ArrayList<Float>(estimatedSize)
            resampleAndFeed(source, count, sourceRate) { sample ->
                temp.add(sample)
            }
            val result = FloatArray(temp.size)
            for (i in temp.indices) {
                result[i] = temp[i]
            }
            return result
        }
    }
}
