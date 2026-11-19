package com.echoshield.echonode.service

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class YamnetLabelsTest {
    @Test fun firearmClassesContributeEvidence() {
        listOf("Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire").forEach {
            assertTrue(it, YamnetLabels.isGunshot(it))
        }
    }

    @Test fun confusableSoundsDoNotBecomeGunshotEvidence() {
        listOf("Bang", "Explosion", "Cap gun", "Firecracker", "Burst, pop", "Pop music", "Gunshot").forEach {
            assertFalse(it, YamnetLabels.isGunshot(it))
        }
    }
}
