package com.echoshield.echonode.service

import android.content.Context
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder

class LegacyTfLiteGunshotClassifier(
    private val context: Context
) {
    companion object {
        private const val TAG = "LegacyTfLiteClassifier"
        private const val MODEL_FILE = "yamnet.tflite"
        private const val LABEL_FILE = "yamnet_class_map.csv"
        const val INPUT_SAMPLES = 15600
    }

    data class Result(
        val gunshotConfidence: Float,
        val topLabel: String,
        val topScore: Float
    )

    private var interpreter: Interpreter? = null
    private var classLabels: List<String> = emptyList()
    private var gunshotIndices: List<Int> = emptyList()
    private var scoreOutputIndex = -1
    private var inputBuffer: ByteBuffer? = null
    private var scoreBuffer: ByteBuffer? = null

    @Synchronized
    fun initialize(): Boolean {
        close()
        return try {
            classLabels = loadLabels()
            if (classLabels.isEmpty()) {
                Log.e(TAG, "Failed loading labels from $LABEL_FILE")
                return false
            }
            gunshotIndices = classLabels.mapIndexedNotNull { index, label ->
                if (YamnetLabels.isGunshot(label)) index else null
            }
            require(gunshotIndices.isNotEmpty()) { "No firearm classes in label map" }

            val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
            val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size)
                .order(ByteOrder.nativeOrder())
            modelBuffer.put(modelBytes)
            modelBuffer.rewind()

            val options = Interpreter.Options().apply {
                setNumThreads(2)
            }
            val loaded = Interpreter(modelBuffer, options)
            interpreter = loaded
            val inputTensor = loaded.getInputTensor(0)
            require(inputTensor.dataType() == DataType.FLOAT32 &&
                inputTensor.numElements() == INPUT_SAMPLES) { "Unsupported YAMNet input" }
            scoreOutputIndex = (0 until loaded.outputTensorCount).single { index ->
                val tensor = loaded.getOutputTensor(index)
                tensor.dataType() == DataType.FLOAT32 &&
                    tensor.shape().lastOrNull() == classLabels.size
            }
            require(loaded.getOutputTensor(scoreOutputIndex).numElements() == classLabels.size) {
                "Expected one YAMNet score frame"
            }
            inputBuffer = ByteBuffer.allocateDirect(INPUT_SAMPLES * 4).order(ByteOrder.nativeOrder())
            scoreBuffer = ByteBuffer.allocateDirect(classLabels.size * 4).order(ByteOrder.nativeOrder())

            Log.i(TAG, "Legacy TFLite initialized labels=${classLabels.size} gunIndices=${gunshotIndices.size}")
            true
        } catch (e: Exception) {
            close()
            Log.e(TAG, "Failed to initialize legacy TensorFlow Lite classifier", e)
            false
        }
    }

    @Synchronized
    fun classify(audio16k: FloatArray): Result? {
        val activeInterpreter = interpreter ?: return null
        return try {
            require(audio16k.size == INPUT_SAMPLES && audio16k.all { it.isFinite() }) {
                "Expected $INPUT_SAMPLES finite PCM samples"
            }
            // Raw buffers work for either [15600] or [1, 15600] waveform tensors.
            // Calls are serialized, so reuse native buffers instead of allocating
            // roughly 64 KB of direct memory on every inference.
            val input = requireNotNull(inputBuffer)
            input.clear()
            input.asFloatBuffer().put(audio16k)
            val output = requireNotNull(scoreBuffer)
            output.clear()
            activeInterpreter.runForMultipleInputsOutputs(
                arrayOf(input), mutableMapOf<Int, Any>(scoreOutputIndex to output)
            )
            output.rewind()
            val scores = FloatArray(classLabels.size)
            output.asFloatBuffer().get(scores)
            require(scores.all { it.isFinite() && it in 0f..1f }) { "Invalid YAMNet scores" }

            var topIndex = 0
            var topScore = scores[0]
            for (i in 1 until scores.size) {
                if (scores[i] > topScore) {
                    topScore = scores[i]
                    topIndex = i
                }
            }

            var gunshotScore = 0f
            var bestGunshotIdx = -1
            gunshotIndices.forEach { idx ->
                if (idx in scores.indices && scores[idx] > gunshotScore) {
                    gunshotScore = scores[idx]
                    bestGunshotIdx = idx
                }
            }
            if (gunshotScore > 0.05f) {
                Log.d(
                    TAG,
                    "Legacy TF gunshot=${"%.3f".format(gunshotScore)} label=${
                        classLabels.getOrElse(bestGunshotIdx) { "?" }
                    }"
                )
            }

            Result(
                gunshotConfidence = gunshotScore,
                topLabel = classLabels.getOrElse(topIndex) { "unknown" },
                topScore = topScore
            )
        } catch (e: Exception) {
            Log.e(TAG, "Legacy TF inference failed", e)
            null
        }
    }

    @Synchronized
    fun close() {
        interpreter?.close()
        interpreter = null
        scoreOutputIndex = -1
        inputBuffer = null
        scoreBuffer = null
    }

    private fun loadLabels(): List<String> {
        val labels = mutableListOf<String>()
        try {
            context.assets.open(LABEL_FILE).use { stream ->
                BufferedReader(InputStreamReader(stream)).useLines { lines ->
                    lines.drop(1).forEach { line ->
                        // The display name can contain commas ("Gunshot, gunfire").
                        val parts = line.split(",", limit = 3)
                        require(parts.size == 3 && parts[0].toInt() == labels.size) {
                            "Invalid or out-of-order YAMNet label row"
                        }
                        labels += parts[2].trim().removeSurrounding("\"")
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load labels", e)
            return emptyList()
        }
        return labels
    }
}
