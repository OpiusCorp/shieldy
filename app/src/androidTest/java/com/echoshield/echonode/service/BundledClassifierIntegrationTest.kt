package com.echoshield.echonode.service

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Runs the actual packaged asset through the Android TFLite runtime. */
@RunWith(AndroidJUnit4::class)
class BundledClassifierIntegrationTest {
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
