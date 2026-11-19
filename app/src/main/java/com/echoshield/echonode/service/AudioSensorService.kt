package com.echoshield.echonode.service

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.echoshield.echonode.MainActivity
import com.echoshield.echonode.R
import com.echoshield.echonode.core.Triangulation
import com.echoshield.echonode.data.MeshNetworkManager
import com.echoshield.echonode.data.SystemEventFlow
import com.echoshield.echonode.sensor.LocationProvider
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.sqrt

class AudioSensorService : Service() {

    companion object {
        private const val TAG = "AudioSensorService"
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "echoshield_sensor_channel"

        // ─────────────────────────────────────────────────────────────────────
        // AUDIO CAPTURE SETTINGS
        // ─────────────────────────────────────────────────────────────────────
        private val CANDIDATE_SAMPLE_RATES = intArrayOf(16000, 44100, 48000)
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        private const val BYTES_PER_SAMPLE = 2

        // ─────────────────────────────────────────────────────────────────────
        // TWO-TIER DETECTOR TUNING CONSTANTS
        // ─────────────────────────────────────────────────────────────────────
        
        // TIER 1: Low-power activity gate
        // EMA alpha: higher = more responsive to spikes, lower = more stable.
        private const val EMA_ALPHA = 0.4
        
        // PEAK detection: Gunshots are impulsive sounds with high peak-to-RMS ratio.
        private const val PEAK_THRESHOLD_NORMALIZED = 0.25f
        
        // If peak/RMS ratio is high, the sound is likely impulsive (like a gunshot)
        private const val IMPULSIVE_PEAK_RMS_RATIO = 3.0f

        // CLIPPING: Cheap phone mics saturate on nearby gunshots (samples pin to
        // ±32767). A sample is "clipped" when its magnitude is within CLIP_MARGIN of
        // full scale.
        private const val CLIP_MARGIN = 64                 // within ~0.2% of full scale
        private const val CLIP_FRACTION_GATE = 0.005f      // 0.5% of samples clipped
        private const val CLIP_FRACTION_SEVERE = 0.02f     // 2% of samples clipped
        // ML threshold applied when audio is severely clipped (lower = more sensitive).
        private const val CLIPPED_CONFIDENCE_THRESHOLD = 0.06f

        // TIER 2: ML confirmation
        // Gunshot confidence threshold. YAMNet typically outputs 0.0-1.0 for each class.
        private const val GUNSHOT_CONFIDENCE_THRESHOLD = 0.10f
        // Minimum consecutive high-confidence frames before trigger.
        private const val MIN_CONSECUTIVE_DETECTIONS = 1
        
        // ML inference rate limit. Running inference too often drains battery.
        private const val MODEL_INFERENCE_INTERVAL_MS = 400L

        // COOLDOWN: Prevent rapid re-triggers after a detection.
        private const val TRIGGER_COOLDOWN_MS = 5000L
        
        // Telemetry update rate for UI. 100ms = 10 updates/sec.
        private const val AMPLITUDE_UPDATE_INTERVAL_MS = 100L

        // ─────────────────────────────────────────────────────────────────────
        // MODEL SETTINGS (TFLite YAMNet expects 16kHz input, ~0.975s window)
        // ─────────────────────────────────────────────────────────────────────
        private const val MODEL_SAMPLE_RATE = 16000
        private const val MODEL_INPUT_SAMPLES = LegacyTfLiteGunshotClassifier.INPUT_SAMPLES

        // ─────────────────────────────────────────────────────────────────────
        // RETROSPECTIVE AUDIO HISTORY & TIMING
        // ─────────────────────────────────────────────────────────────────────
        private const val HISTORY_SECONDS = 4
        private const val HISTORY_SAMPLES = MODEL_SAMPLE_RATE * HISTORY_SECONDS

        // How recently this node must have heard an impulse of its own for that to
        // count as its arrival of the event a peer is asking about.
        private const val IMPULSE_MEMORY_MS = 3000L

        // Peak level that marks a frame as containing an impulsive onset worth
        // timestamping for triangulation.
        private const val IMPULSE_ONSET_PEAK_NORMALIZED = 0.15f

        // Audio kept after the impulse onset in the classified window to capture
        // reverberant decay.
        private const val WINDOW_TAIL_MS = 300L

        // Maximum duration to wait for tail accumulation before forcing classification
        // so that sustained noise or rapid impulses do not indefinitely postpone inference.
        private const val MAX_TAIL_WAIT_MS = 450L

        fun startService(context: Context) {
            val intent = Intent(context, AudioSensorService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun stopService(context: Context) {
            val intent = Intent(context, AudioSensorService::class.java)
            context.stopService(intent)
        }
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var audioRecordingJob: Job? = null
    private var audioRecord: AudioRecord? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private lateinit var meshNetworkManager: MeshNetworkManager
    private lateinit var legacyTfClassifier: LegacyTfLiteGunshotClassifier
    private var locationProvider: LocationProvider? = null
    @Volatile
    private var legacyTfReady = false
    
    // Two-tier detector state
    private var lastTriggerTime = 0L
    private var smoothedAmplitude = 0.0
    private var consecutiveHighConfidence = 0
    private var currentGateOpen = false

    // Sentinel duty state - controls whether this device acts as sentinel or pre-roll listener
    @Volatile
    private var isSentinelActive = true
    @Volatile
    private var isProcessingWakeRequest = false
    @Volatile
    private var pendingWakeSessionId: String? = null
    private var dutyMonitorJob: Job? = null
    private var wakeRequestJob: Job? = null
    
    // Thread-safe circular audio buffer, resampler, and detection scheduler
    private val rollingAudioBuffer = RollingAudioBuffer(HISTORY_SAMPLES, MODEL_SAMPLE_RATE)
    private val detectionScheduler = DetectionScheduler(
        windowTailMs = WINDOW_TAIL_MS,
        maxTailWaitMs = MAX_TAIL_WAIT_MS,
        minInferenceIntervalMs = MODEL_INFERENCE_INTERVAL_MS
    )
    @Volatile
    private var actualSampleRate: Int = MODEL_SAMPLE_RATE

    /** Wall-clock time this node's microphone last heard an impulsive onset. */
    @Volatile
    private var lastImpulseAtMs = 0L

    /** Set once per wake request so a filling buffer cannot fire a burst of votes. */
    @Volatile
    private var wakeVoteDispatched = false

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service created")
        createNotificationChannel()
        acquireWakeLock()
        meshNetworkManager = MeshNetworkManager.getInstance(applicationContext)
        legacyTfClassifier = LegacyTfLiteGunshotClassifier(applicationContext)
        locationProvider = LocationProvider(applicationContext)

        serviceScope.launch {
            legacyTfReady = legacyTfClassifier.initialize()
            Log.i(TAG, "Legacy TF classifier initialized: $legacyTfReady")
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "Service started")
        
        startForegroundWithNotification()
        meshNetworkManager.startMesh()
        startDutyMonitoring()
        startWakeRequestHandling()
        
        // Start continuous audio recording for rolling pre-roll buffer
        val assignment = meshNetworkManager.dutyAssignment.value
        isSentinelActive = assignment.audioSentinelDuty
        if (isSentinelActive) {
            Log.i(TAG, "Starting as SENTINEL - active detection enabled")
        } else {
            Log.i(TAG, "Starting as NON-SENTINEL - pre-roll recording active (low CPU)")
        }
        startAudioRecording()
        
        SystemEventFlow.setServiceRunning(true)
        
        return START_STICKY
    }

    private fun startDutyMonitoring() {
        if (dutyMonitorJob?.isActive == true) return

        dutyMonitorJob = serviceScope.launch(Dispatchers.Main) {
            meshNetworkManager.dutyAssignment.collect { assignment ->
                val wasActive = isSentinelActive
                isSentinelActive = assignment.audioSentinelDuty

                if (wasActive != isSentinelActive) {
                    Log.i(TAG, "Sentinel duty changed: active=$isSentinelActive (leader=${assignment.sentinelNodeId})")
                    
                    if (isSentinelActive) {
                        locationProvider?.startLocationUpdates()
                    } else {
                        locationProvider?.stopLocationUpdates()
                        currentGateOpen = false
                        consecutiveHighConfidence = 0
                        SystemEventFlow.updateGateOpen(false)
                    }
                    // Audio capture remains running in the background into rollingAudioBuffer
                    // so that non-sentinels retain pre-roll history for peer wake requests.
                    if (audioRecordingJob?.isActive != true) {
                        startAudioRecording()
                    }
                }
            }
        }
    }

    private fun startWakeRequestHandling() {
        if (wakeRequestJob?.isActive == true) return

        wakeRequestJob = serviceScope.launch(Dispatchers.Main) {
            meshNetworkManager.wakeClassifyRequests.collect { request ->
                Log.i(TAG, "Received WAKE_CLASSIFY request: session=${request.sessionId}")
                handleWakeClassifyRequest(request.sessionId)
            }
        }
    }

    private fun handleWakeClassifyRequest(sessionId: String) {
        if (isProcessingWakeRequest) {
            Log.d(TAG, "Already processing wake request, skipping")
            return
        }

        // Already recording with enough pre-roll: answer from retained history immediately.
        if (audioRecordingJob?.isActive == true && rollingAudioBuffer.hasEnoughSamples(MODEL_INPUT_SAMPLES)) {
            classifyAndVote(sessionId)
            return
        }

        // Not yet enough history: mark pending and vote once window fills.
        isProcessingWakeRequest = true
        wakeVoteDispatched = false
        pendingWakeSessionId = sessionId

        Log.i(TAG, "Waiting for pre-roll buffer to fill for session=$sessionId")
        locationProvider?.startLocationUpdates()
        if (audioRecordingJob?.isActive != true) {
            startAudioRecording()
        }
    }

    /**
     * Classify this node's own audio for a peer's detection session and vote.
     * Retains exact confidence, altitude, and arrival metadata.
     */
    private fun classifyAndVote(sessionId: String) {
        serviceScope.launch(Dispatchers.IO) {
            try {
                isProcessingWakeRequest = true

                val voterLocation = locationProvider?.currentLocation?.value
                val voterLat = voterLocation?.latitude ?: Double.NaN
                val voterLon = voterLocation?.longitude ?: Double.NaN
                val voterAlt = voterLocation?.altitude ?: Double.NaN

                val now = System.currentTimeMillis()
                val impulseAt = lastImpulseAtMs
                val heardImpulse = impulseAt > 0L && now - impulseAt <= IMPULSE_MEMORY_MS
                val windowEnd = if (heardImpulse) impulseAt + WINDOW_TAIL_MS else now
                val arrivalAtMs = if (heardImpulse) impulseAt else Triangulation.NO_TIMING

                // If the requested tail window is in the future relative to captured audio, wait for tail arrival
                if (windowEnd > now) {
                    val maxWaitMs = minOf(MAX_TAIL_WAIT_MS, (windowEnd - now) + 100L)
                    val waitStart = System.currentTimeMillis()
                    while (rollingAudioBuffer.lastWriteEndTimeMs < windowEnd &&
                           System.currentTimeMillis() - waitStart < maxWaitMs &&
                           audioRecordingJob?.isActive == true) {
                        delay(20)
                    }
                }

                if (!legacyTfReady) {
                    Log.w(TAG, "No classifier ready for wake vote, voting NO for $sessionId")
                    meshNetworkManager.submitClassifyVote(
                        sessionId, false, 0f, voterLat, voterLon, voterAlt, Triangulation.NO_TIMING
                    )
                    return@launch
                }

                val modelInput = rollingAudioBuffer.snapshotWindowAt(windowEnd, MODEL_INPUT_SAMPLES)
                if (modelInput == null) {
                    Log.w(TAG, "No audio history available for requested window ($windowEnd) in $sessionId, voting NO")
                    meshNetworkManager.submitClassifyVote(
                        sessionId, false, 0f, voterLat, voterLon, voterAlt, Triangulation.NO_TIMING
                    )
                    return@launch
                }

                val result = classifyTfLite(modelInput)

                if (result != null) {
                    val isGunshot = result.gunshotConfidence >= GUNSHOT_CONFIDENCE_THRESHOLD
                    Log.i(
                        TAG,
                        "Wake classify result: gunshot=$isGunshot score=${result.gunshotConfidence} " +
                            "top=${result.topLabel} windowEnd=${now - windowEnd}ms ago heardImpulse=$heardImpulse"
                    )
                    meshNetworkManager.submitClassifyVote(
                        sessionId = sessionId,
                        isGunshot = isGunshot,
                        confidence = result.gunshotConfidence,
                        latitude = voterLat,
                        longitude = voterLon,
                        altitude = voterAlt,
                        detectedAtMs = if (isGunshot) arrivalAtMs else Triangulation.NO_TIMING
                    )
                    SystemEventFlow.updateModelInference(result.gunshotConfidence, result.topLabel)
                } else {
                    Log.w(TAG, "Classifier returned null for $sessionId")
                    meshNetworkManager.submitClassifyVote(
                        sessionId, false, 0f, voterLat, voterLon, voterAlt, Triangulation.NO_TIMING
                    )
                }
            } finally {
                isProcessingWakeRequest = false
                pendingWakeSessionId = null
                if (!isSentinelActive) {
                    locationProvider?.stopLocationUpdates()
                }
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        Log.d(TAG, "Service destroyed")
        dutyMonitorJob?.cancel()
        dutyMonitorJob = null
        wakeRequestJob?.cancel()
        wakeRequestJob = null
        stopAudioRecording()
        meshNetworkManager.stopMesh()
        legacyTfClassifier.close()
        locationProvider?.stopLocationUpdates()
        releaseWakeLock()
        serviceScope.cancel()
        SystemEventFlow.setServiceRunning(false)
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = getString(R.string.notification_channel_description)
                setShowBadge(false)
                lockscreenVisibility = Notification.VISIBILITY_PUBLIC
            }
            
            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun startForegroundWithNotification() {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(getString(R.string.notification_text))
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            )
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun acquireWakeLock() {
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "EchoShield::AudioSensorWakeLock"
        ).apply {
            acquire(60 * 60 * 1000L) // 1 hour max
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.let {
            if (it.isHeld) {
                it.release()
            }
        }
        wakeLock = null
    }

    private data class AudioRecordConfig(
        val audioRecord: AudioRecord,
        val sampleRate: Int,
        val bufferSizeSamples: Int
    )

    private fun initializeAudioRecord(): AudioRecordConfig? {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) !=
            PackageManager.PERMISSION_GRANTED) {
            Log.w(TAG, "Microphone permission unavailable; audio capture not started")
            return null
        }
        for (rate in CANDIDATE_SAMPLE_RATES) {
            val minBufferSizeBytes = AudioRecord.getMinBufferSize(rate, CHANNEL_CONFIG, AUDIO_FORMAT)
            if (minBufferSizeBytes <= 0 || minBufferSizeBytes == AudioRecord.ERROR || minBufferSizeBytes == AudioRecord.ERROR_BAD_VALUE) {
                Log.w(TAG, "AudioRecord minBufferSize invalid for rate $rate: $minBufferSizeBytes")
                continue
            }

            val minBufferSizeSamples = minBufferSizeBytes / BYTES_PER_SAMPLE
            val recordBufferSizeBytes = minBufferSizeBytes * 2

            try {
                val record = AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    rate,
                    CHANNEL_CONFIG,
                    AUDIO_FORMAT,
                    recordBufferSizeBytes
                )
                if (record.state == AudioRecord.STATE_INITIALIZED) {
                    Log.i(TAG, "AudioRecord initialized at ${rate}Hz (bufferBytes=$minBufferSizeBytes, bufferSamples=$minBufferSizeSamples)")
                    return AudioRecordConfig(record, rate, minBufferSizeSamples)
                } else {
                    Log.w(TAG, "AudioRecord not initialized at ${rate}Hz")
                    record.release()
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to create AudioRecord at ${rate}Hz: ${e.message}")
            }
        }
        return null
    }

    private fun startAudioRecording() {
        if (audioRecordingJob?.isActive == true) {
            Log.d(TAG, "Audio recording already active")
            return
        }

        val config = initializeAudioRecord()
        if (config == null) {
            Log.e(TAG, "Failed to initialize AudioRecord with any candidate sample rate")
            return
        }

        audioRecord = config.audioRecord
        actualSampleRate = config.sampleRate
        val bufferSizeSamples = config.bufferSizeSamples

        try {
            audioRecord?.startRecording()
            Log.d(TAG, "Audio recording started at ${actualSampleRate}Hz with buffer size: $bufferSizeSamples samples")

            audioRecordingJob = serviceScope.launch {
                processAudioStream(bufferSizeSamples)
            }
        } catch (e: SecurityException) {
            Log.e(TAG, "Security exception starting audio recording", e)
        } catch (e: Exception) {
            Log.e(TAG, "Exception starting audio recording", e)
        }
    }

    private suspend fun processAudioStream(bufferSizeSamples: Int) {
        val buffer = ShortArray(bufferSizeSamples)
        var lastAmplitudeUpdate = 0L

        if (isSentinelActive) {
            locationProvider?.startLocationUpdates()
        }

        while (audioRecordingJob?.isActive == true) {
            val readResult = audioRecord?.read(buffer, 0, buffer.size) ?: -1

            if (readResult > 0) {
                val currentTime = System.currentTimeMillis()

                val instantRms = calculateRMS(buffer, readResult)
                val peakIndex = indexOfMaxAmplitude(buffer, readResult)
                val peakAmplitude = abs(buffer[peakIndex].toInt())
                val peakNormalized = peakAmplitude / 32768f
                val clipFraction = calculateClipFraction(buffer, readResult)
                val clippingGate = clipFraction >= CLIP_FRACTION_GATE
                val clippingSevere = clipFraction >= CLIP_FRACTION_SEVERE

                // Feed rolling pre-roll buffer using native write or stateful resampler
                if (actualSampleRate == MODEL_SAMPLE_RATE) {
                    rollingAudioBuffer.write(buffer, readResult, currentTime)
                } else {
                    rollingAudioBuffer.writeResampled(buffer, readResult, actualSampleRate, currentTime)
                }

                // Timestamp the impulse itself to sample accuracy using actual capture rate
                val detectedImpulseOnsetMs: Long? = if (peakNormalized >= IMPULSE_ONSET_PEAK_NORMALIZED || clippingGate) {
                    val frameStartMs = currentTime - (readResult * 1000L / actualSampleRate)
                    val onset = frameStartMs + (peakIndex * 1000L / actualSampleRate)
                    lastImpulseAtMs = onset
                    onset
                } else {
                    null
                }

                // A woken non-sentinel votes once enough history is buffered
                if (isProcessingWakeRequest && !isSentinelActive) {
                    val sessionId = pendingWakeSessionId
                    if (!wakeVoteDispatched && sessionId != null && rollingAudioBuffer.hasEnoughSamples(MODEL_INPUT_SAMPLES)) {
                        wakeVoteDispatched = true
                        Log.i(TAG, "Wake capture window filled, classifying for session=$sessionId")
                        classifyAndVote(sessionId)
                    }
                    continue
                }

                if (!isSentinelActive) {
                    // Non-sentinels retain pre-roll history only (low CPU/battery)
                    continue
                }

                // TIER 1: Apply EMA smoothing for stable gate decisions
                smoothedAmplitude = EMA_ALPHA * instantRms + (1 - EMA_ALPHA) * smoothedAmplitude

                val cooldownRemaining = maxOf(0L, TRIGGER_COOLDOWN_MS - (currentTime - lastTriggerTime))
                
                // Update telemetry at fixed interval
                if (currentTime - lastAmplitudeUpdate >= AMPLITUDE_UPDATE_INTERVAL_MS) {
                    SystemEventFlow.updateAmplitude(instantRms)
                    SystemEventFlow.updateSmoothedAmplitude(smoothedAmplitude)
                    SystemEventFlow.updateCooldownRemaining(cooldownRemaining)
                    SystemEventFlow.updateClipping(clippingGate, clipFraction)
                    lastAmplitudeUpdate = currentTime
                }

                // TIER 1: Activity gate check
                val gateThreshold = SystemEventFlow.detectionThreshold.value
                val rmsGateOpen = instantRms >= gateThreshold
                val peakGateOpen = peakNormalized >= PEAK_THRESHOLD_NORMALIZED
                val impulsiveSound = instantRms > 50 && (peakAmplitude / instantRms) > IMPULSIVE_PEAK_RMS_RATIO

                val gateOpen = (rmsGateOpen || peakGateOpen || impulsiveSound || clippingGate) &&
                    rollingAudioBuffer.hasEnoughSamples(MODEL_INPUT_SAMPLES)

                // Latch pending event through gate closure and tail collection
                detectionScheduler.onFrame(currentTime, gateOpen, detectedImpulseOnsetMs)
                
                if (gateOpen != currentGateOpen) {
                    currentGateOpen = gateOpen
                    SystemEventFlow.updateGateOpen(gateOpen)
                    if (gateOpen) {
                        val reason = when {
                            rmsGateOpen -> "RMS"
                            peakGateOpen -> "PEAK"
                            impulsiveSound -> "IMPULSIVE"
                            clippingGate -> "CLIP"
                            else -> "?"
                        }
                        Log.d(TAG, "Activity gate OPENED [$reason] rms=${String.format("%.1f", instantRms)} peak=${"%.3f".format(peakNormalized)} clip=${"%.3f".format(clipFraction)} thresh=$gateThreshold")
                    } else {
                        Log.d(TAG, "Activity gate CLOSED")
                        consecutiveHighConfidence = 0
                    }
                }

                // TIER 2: Bounded post-impulse tail accumulation via scheduler
                val shouldRunModel = detectionScheduler.shouldRunInference(currentTime, legacyTfReady)

                if (shouldRunModel) {
                    val windowEnd = detectionScheduler.getTargetWindowEndMs(currentTime)
                    val modelInput = rollingAudioBuffer.snapshotWindowAt(windowEnd, MODEL_INPUT_SAMPLES)

                    if (modelInput != null) {
                        val result = classifyTfLite(modelInput)
                        detectionScheduler.onInferenceCompleted(currentTime)

                        if (result != null) {
                            val effectiveThreshold =
                                if (clippingSevere) CLIPPED_CONFIDENCE_THRESHOLD else GUNSHOT_CONFIDENCE_THRESHOLD
                            val isAboveThreshold = result.gunshotConfidence >= effectiveThreshold
                            Log.d(TAG, "ML: score=${"%.3f".format(result.gunshotConfidence)} " +
                                "(thresh=$effectiveThreshold clip=${"%.3f".format(clipFraction)}, above=$isAboveThreshold) " +
                                "top=${result.topLabel}@${"%.3f".format(result.topScore)}")
                            
                            SystemEventFlow.updateModelInference(
                                gunshotConfidence = result.gunshotConfidence,
                                topLabel = result.topLabel
                            )

                            if (isAboveThreshold) {
                                consecutiveHighConfidence++
                                Log.i(TAG, "HIGH CONFIDENCE FRAME #$consecutiveHighConfidence")
                            } else {
                                consecutiveHighConfidence = 0
                            }

                            if (consecutiveHighConfidence >= MIN_CONSECUTIVE_DETECTIONS &&
                                cooldownRemaining == 0L) {
                                Log.w(
                                    TAG,
                                    "SENTINEL DETECTED THREAT! score=${result.gunshotConfidence} " +
                                    "consecutive=$consecutiveHighConfidence top=${result.topLabel}"
                                )
                                lastTriggerTime = currentTime
                                consecutiveHighConfidence = 0
                                
                                val location = locationProvider?.currentLocation?.value
                                val lat = location?.latitude ?: 0.0
                                val lon = location?.longitude ?: 0.0
                                val alt = location?.altitude ?: Double.NaN

                                val connectedPeers = meshNetworkManager.getConnectedPeerIds().size
                                if (connectedPeers > 0) {
                                    meshNetworkManager.disarmSentinel()
                                } else {
                                    Log.i(TAG, "Threat detected with no peers; keeping local sentinel active")
                                }

                                val impulseAt = lastImpulseAtMs
                                val arrivalAtMs = if (impulseAt > 0L && currentTime - impulseAt <= IMPULSE_MEMORY_MS) {
                                    impulseAt
                                } else {
                                    currentTime
                                }

                                meshNetworkManager.broadcastWakeClassify(
                                    latitude = lat,
                                    longitude = lon,
                                    confidence = result.gunshotConfidence,
                                    altitude = alt,
                                    detectedAtMs = arrivalAtMs
                                )

                                Log.i(
                                    TAG,
                                    "WAKE_CLASSIFY broadcast at ($lat, $lon), peers=$connectedPeers " +
                                        "conf=${result.gunshotConfidence} arrival=${currentTime - arrivalAtMs}ms ago"
                                )
                            }
                        }
                    }
                }
            } else if (readResult == AudioRecord.ERROR_INVALID_OPERATION) {
                Log.e(TAG, "AudioRecord read error: invalid operation")
                delay(100)
            } else if (readResult == AudioRecord.ERROR_BAD_VALUE) {
                Log.e(TAG, "AudioRecord read error: bad value")
                delay(100)
            }
        }
    }

    private fun calculateRMS(buffer: ShortArray, readSize: Int): Double {
        var sum = 0.0
        for (i in 0 until readSize) {
            sum += buffer[i].toDouble() * buffer[i].toDouble()
        }
        return sqrt(sum / readSize)
    }

    private fun calculateClipFraction(buffer: ShortArray, readSize: Int): Float {
        if (readSize <= 0) return 0f
        val clipHigh = Short.MAX_VALUE - CLIP_MARGIN
        val clipLow = Short.MIN_VALUE + CLIP_MARGIN
        var clipped = 0
        for (i in 0 until readSize) {
            val sample = buffer[i].toInt()
            if (sample >= clipHigh || sample <= clipLow) {
                clipped++
            }
        }
        return clipped.toFloat() / readSize.toFloat()
    }

    private fun indexOfMaxAmplitude(buffer: ShortArray, readSize: Int): Int {
        var maxAmplitude = -1
        var maxIndex = 0
        for (i in 0 until readSize) {
            val amplitude = abs(buffer[i].toInt())
            if (amplitude > maxAmplitude) {
                maxAmplitude = amplitude
                maxIndex = i
            }
        }
        return maxIndex
    }

    private data class ClassificationResult(
        val gunshotConfidence: Float,
        val topLabel: String,
        val topScore: Float
    )

    private fun classifyTfLite(modelInput: FloatArray): ClassificationResult? {
        val legacy = if (legacyTfReady) legacyTfClassifier.classify(modelInput) else null
        if (legacy == null) return null
        return ClassificationResult(
            gunshotConfidence = legacy.gunshotConfidence.coerceIn(0f, 1f),
            topLabel = legacy.topLabel,
            topScore = legacy.topScore
        )
    }

    private fun stopAudioRecording() {
        audioRecordingJob?.cancel()
        audioRecordingJob = null

        audioRecord?.let {
            try {
                if (it.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    it.stop()
                }
                it.release()
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping audio recording", e)
            }
        }
        audioRecord = null
        
        smoothedAmplitude = 0.0
        consecutiveHighConfidence = 0
        currentGateOpen = false
        
        rollingAudioBuffer.reset()
        detectionScheduler.reset()
        lastImpulseAtMs = 0L
        wakeVoteDispatched = false
        
        SystemEventFlow.updateAmplitude(0.0)
        SystemEventFlow.updateSmoothedAmplitude(0.0)
        SystemEventFlow.updateGateOpen(false)
        SystemEventFlow.updateCooldownRemaining(0L)
        SystemEventFlow.updateModelInference(0f, "idle")
        SystemEventFlow.updateClipping(false, 0f)
        Log.d(TAG, "Audio recording stopped")
    }
}
