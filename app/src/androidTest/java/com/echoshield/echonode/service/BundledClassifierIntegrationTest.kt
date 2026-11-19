package com.echoshield.echonode.service

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.PI
import kotlin.math.sin

/** Runs the actual packaged asset through the Android TFLite runtime. */
@RunWith(AndroidJUnit4::class)
class BundledClassifierIntegrationTest {
    @Test fun packagedModelRecognizesKnownWaveformAfterBufferReuse() {
        val classifier = LegacyTfLiteGunshotClassifier(
            InstrumentationRegistry.getInstrumentation().targetContext
        )
        try {
            assertTrue(classifier.initialize())
            assertNotNull(classifier.classify(FloatArray(LegacyTfLiteGunshotClassifier.INPUT_SAMPLES)))
            val tone = FloatArray(LegacyTfLiteGunshotClassifier.INPUT_SAMPLES) { index ->
                sin(2.0 * PI * 440.0 * index / 16000.0).toFloat()
            }
            val result = classifier.classify(tone)
            assertNotNull(result)
            assertEquals("Sine wave", result!!.topLabel)
            assertTrue("Scores must reflect the new waveform", result.topScore > 0.8f)
        } finally {
            classifier.close()
        }
    }

    @Test fun packagedModelInitializesAndClassifiesRepeatedWindows() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val classifier = LegacyTfLiteGunshotClassifier(context)
        try {
            assertTrue("Bundled model and label contract must initialize", classifier.initialize())
            repeat(3) {
                val result = classifier.classify(FloatArray(LegacyTfLiteGunshotClassifier.INPUT_SAMPLES))
                assertNotNull("Actual TFLite inference must return scores", result)
                assertTrue(result!!.gunshotConfidence.isFinite())
                assertTrue(result.gunshotConfidence in 0f..1f)
                assertTrue(result.topScore in 0f..1f)
                assertTrue(result.topLabel.isNotBlank())
            }
            assertTrue("Interpreter can be closed and reinitialized", classifier.initialize())
            assertNotNull(classifier.classify(FloatArray(LegacyTfLiteGunshotClassifier.INPUT_SAMPLES)))
        } finally {
            classifier.close()
        }
    }
}
