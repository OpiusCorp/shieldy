package com.echoshield.echonode.service

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class DetectionSchedulerTest {

    private lateinit var scheduler: DetectionScheduler

    @Before
    fun setUp() {
        scheduler = DetectionScheduler(
            windowTailMs = 300L,
            maxTailWaitMs = 450L,
            minInferenceIntervalMs = 400L,
            eventLifecycleMs = 1500L
        )
    }

    @Test
    fun testImpulseThenSilenceLatchesPendingState() {
        // At t=1000ms: impulse frame arrives, gate opens
        scheduler.onFrame(currentTimeMs = 1000L, gateOpen = true, impulseOnsetMs = 980L)
        assertTrue(scheduler.isPending)
        assertEquals(980L, scheduler.pendingFirstOnsetMs)
        assertEquals(980L, scheduler.pendingLatestOnsetMs)

        // Immediately following frames are quiet (gate closed)
        scheduler.onFrame(currentTimeMs = 1050L, gateOpen = false, impulseOnsetMs = null)
        assertTrue("Pending state must latch across quiet frames", scheduler.isPending)
        assertFalse("Tail not yet ready at 1050ms (< 980 + 300 = 1280ms)", scheduler.isTailReady(1050L))

        // Advance to t=1280ms: tail target reached
        assertTrue(scheduler.isTailReady(1280L))
        assertTrue(scheduler.shouldRunInference(currentTimeMs = 1280L, classifierReady = true))
        assertEquals(1280L, scheduler.getTargetWindowEndMs(1280L))

        // Complete inference
        scheduler.onInferenceCompleted(1280L)
        assertFalse(scheduler.isPending)
        assertEquals(1280L, scheduler.lastInferenceTimeMs)
    }

    @Test
    fun testSustainedImpulsesTimeoutAnchoredToFirstOnset() {
        // First impulse at t=2000ms
        scheduler.onFrame(currentTimeMs = 2000L, gateOpen = true, impulseOnsetMs = 2000L)
        assertEquals(2000L, scheduler.pendingFirstOnsetMs)

        // Continuous train of impulses every 40ms
        for (step in 1..10) {
            val t = 2000L + (step * 40L)
            scheduler.onFrame(currentTimeMs = t, gateOpen = true, impulseOnsetMs = t)
            assertEquals("First onset must remain anchored to 2000ms", 2000L, scheduler.pendingFirstOnsetMs)
            assertEquals("Latest onset tracks incoming impulse", t, scheduler.pendingLatestOnsetMs)
        }

        // At t=2400ms: latest onset was 2400ms (tail target = 2700ms), but timeout is 2000 + 450 = 2450ms
        assertFalse(scheduler.isTailReady(2440L))

        // At t=2450ms: timeout reached even though tail target (2700ms) has not arrived!
        assertTrue("Timeout must trigger tail readiness at firstOnset + maxTailWaitMs", scheduler.isTailReady(2450L))
        assertTrue(scheduler.shouldRunInference(currentTimeMs = 2450L, classifierReady = true))
    }

    @Test
    fun testMinInferenceIntervalEnforced() {
        // Run first inference at t=1000ms
        scheduler.onFrame(currentTimeMs = 1000L, gateOpen = true, impulseOnsetMs = 1000L)
        scheduler.onInferenceCompleted(1000L)

        // New impulse arrives at t=1200ms (only 200ms elapsed, minInterval=400ms)
        scheduler.onFrame(currentTimeMs = 1200L, gateOpen = true, impulseOnsetMs = 1200L)
        // Advance to t=1350ms (isTailReady is true, but minInterval not met)
        assertFalse("Rate limit must block inference before 400ms interval", scheduler.shouldRunInference(1350L, classifierReady = true))

        // At t=1400ms: minInterval satisfied
        assertTrue(scheduler.shouldRunInference(1400L, classifierReady = true))
    }

    @Test
    fun testStalePendingEventExpiresAfterLifecycle() {
        // Pending onset at t=1000ms
        scheduler.onFrame(currentTimeMs = 1000L, gateOpen = true, impulseOnsetMs = 1000L)
        assertTrue(scheduler.isPending)

        // No inference ran, time advances beyond eventLifecycleMs (1500ms) to t=2600ms
        scheduler.onFrame(currentTimeMs = 2600L, gateOpen = false, impulseOnsetMs = null)
        assertFalse("Stale unclassified pending event must expire after lifecycle window", scheduler.isPending)
    }
}
