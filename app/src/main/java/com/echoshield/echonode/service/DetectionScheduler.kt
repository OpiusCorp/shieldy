package com.echoshield.echonode.service

/**
 * Shared scheduling and lifecycle controller for gunshot detection inference.
 *
 * Solves:
 * 1. Impulse-then-silence latching: latches a pending event upon gate opening or impulse onset,
 *    keeping it eligible for inference through reverberant tail collection even if the instantaneous
 *    gate drops to closed on the next frame.
 * 2. Sustained impulse bounding: anchors timeout to the first pending onset ([pendingFirstOnsetMs])
 *    so continuous noise or repeated impulse trains cannot indefinitely reset the timeout.
 * 3. Event lifecycle: ensures old historical impulses expire after [eventLifecycleMs] and cannot
 *    contaminate subsequent RMS-only gate activations.
 */
class DetectionScheduler(
    val windowTailMs: Long = DEFAULT_WINDOW_TAIL_MS,
    val maxTailWaitMs: Long = DEFAULT_MAX_TAIL_WAIT_MS,
    val minInferenceIntervalMs: Long = DEFAULT_INFERENCE_INTERVAL_MS,
    val eventLifecycleMs: Long = DEFAULT_EVENT_LIFECYCLE_MS
) {
    companion object {
        const val DEFAULT_WINDOW_TAIL_MS = 300L
        const val DEFAULT_MAX_TAIL_WAIT_MS = 450L
        const val DEFAULT_INFERENCE_INTERVAL_MS = 400L
        const val DEFAULT_EVENT_LIFECYCLE_MS = 1500L
    }

    var isPending: Boolean = false
        private set

    var pendingFirstOnsetMs: Long = 0L
        private set

    var pendingLatestOnsetMs: Long = 0L
        private set

    var lastInferenceTimeMs: Long = 0L
        private set

    /**
     * Ingests a new audio frame's gate status and optional impulse onset timestamp.
     */
    fun onFrame(currentTimeMs: Long, gateOpen: Boolean, impulseOnsetMs: Long? = null) {
        // Expire stale pending events if unclassified after lifecycle window
        if (isPending && (currentTimeMs - pendingFirstOnsetMs > eventLifecycleMs)) {
            resetPending()
        }

        if (gateOpen || impulseOnsetMs != null) {
            val onset = impulseOnsetMs ?: currentTimeMs
            if (!isPending) {
                isPending = true
                pendingFirstOnsetMs = onset
                pendingLatestOnsetMs = onset
            } else if (impulseOnsetMs != null) {
                pendingLatestOnsetMs = impulseOnsetMs
            }
        }
    }

    /**
     * Determines whether post-impulse reverberant tail collection is satisfied or timed out.
     */
    fun isTailReady(currentTimeMs: Long): Boolean {
        if (!isPending) return false
        val tailTargetMs = pendingLatestOnsetMs + windowTailMs
        val timeoutMs = pendingFirstOnsetMs + maxTailWaitMs
        return (currentTimeMs >= tailTargetMs) || (currentTimeMs >= timeoutMs)
    }

    /**
     * Determines if Tier 2 ML inference should be executed now.
     */
    fun shouldRunInference(currentTimeMs: Long, classifierReady: Boolean): Boolean {
        if (!classifierReady || !isPending) return false
        if (currentTimeMs - lastInferenceTimeMs < minInferenceIntervalMs) return false
        return isTailReady(currentTimeMs)
    }

    /**
     * Returns the target window end timestamp to classify.
     */
    fun getTargetWindowEndMs(currentTimeMs: Long): Long {
        if (!isPending) return currentTimeMs
        val tailTargetMs = pendingLatestOnsetMs + windowTailMs
        return minOf(currentTimeMs, tailTargetMs)
    }

    /**
     * Call when inference completes to record completion and clear pending state.
     */
    fun onInferenceCompleted(currentTimeMs: Long) {
        lastInferenceTimeMs = currentTimeMs
        resetPending()
    }

    /**
     * Resets scheduler state.
     */
    fun reset() {
        lastInferenceTimeMs = 0L
        resetPending()
    }

    private fun resetPending() {
        isPending = false
        pendingFirstOnsetMs = 0L
        pendingLatestOnsetMs = 0L
    }
}
