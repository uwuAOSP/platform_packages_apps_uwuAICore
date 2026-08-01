/*
 * Copyright (C) 2026 The uwuAOSP Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.uwuaosp.aicore.ocr

import android.app.DownloadManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.os.RemoteCallbackList
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

private object OcrNative {
    init {
        System.loadLibrary("uwu_aicore_ocr_jni")
    }

    fun ensureLoaded() = Unit

    external fun recognize(
        modelPath: String,
        mmprojPath: String,
        imagePath: String,
        threadCount: Int,
        useVulkan: Boolean,
    ): String

    external fun prewarm(
        modelPath: String,
        mmprojPath: String,
        threadCount: Int,
        useVulkan: Boolean,
    ): Boolean

    external fun requestCancel()
    external fun unload()
}

class OcrService : Service() {
    private val callbacks = RemoteCallbackList<IOcrCallback>()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val nativeDispatcher: ExecutorCoroutineDispatcher =
        Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val serviceScope = CoroutineScope(SupervisorJob())
    private val requests = ConcurrentHashMap<Long, Job>()
    @Volatile
    private var activeRequestId = NO_REQUEST
    @Volatile
    private var shuttingDown = false
    @Volatile
    private var nativeLoaded = false
    @Volatile
    private var prewarmJob: Job? = null
    @Volatile
    private var vulkanUnavailable = false
    private lateinit var modelStore: OcrModelStore

    private val unloadRunnable = Runnable {
        if (requests.isEmpty() && nativeLoaded) {
            serviceScope.launch(nativeDispatcher) {
                if (!nativeLoaded) return@launch
                runCatching { OcrNative.unload() }
                    .onSuccess {
                        nativeLoaded = false
                        Log.i(TAG, "Released idle OCR model")
                    }
                    .onFailure { error ->
                        Log.w(TAG, "Could not release OCR model", error)
                    }
            }
        }
    }

    private val binder = object : IOcrService.Stub() {
        override fun registerCallback(callback: IOcrCallback) {
            callbacks.register(callback)
            val snapshot = modelStore.snapshot
            callback.onModelStateChanged(
                snapshot.status,
                snapshot.downloadedBytes,
                snapshot.totalBytes,
            )
        }

        override fun unregisterCallback(callback: IOcrCallback) {
            callbacks.unregister(callback)
        }

        override fun getModelState(): Int = modelStore.snapshot.status

        override fun getDownloadedBytes(): Long = modelStore.snapshot.downloadedBytes

        override fun getDownloadTotalBytes(): Long = modelStore.snapshot.totalBytes

        override fun startDownload() {
            mainHandler.post(modelStore::startDownload)
        }

        override fun cancelDownload() {
            mainHandler.post(modelStore::cancelDownload)
        }

        override fun deleteModels() {
            cancelAllRecognition()
            mainHandler.post {
                modelStore.deleteModels()
                scheduleUnload(0)
            }
        }

        override fun recognize(
            image: ParcelFileDescriptor,
            requestId: Long,
            useVulkan: Boolean,
        ) {
            if (requestId <= 0) {
                image.close()
                notifyError(requestId, ERROR_INVALID_REQUEST, "Invalid request ID")
                return
            }
            if (modelStore.snapshot.status != ModelStatus.READY) {
                image.close()
                notifyError(requestId, ERROR_MODEL_MISSING, "OCR model is not ready")
                return
            }

            mainHandler.removeCallbacks(unloadRunnable)
            val job = serviceScope.launch(
                nativeDispatcher,
                start = CoroutineStart.LAZY,
            ) {
                notifyProgress(requestId, 0)
                val effectiveVulkan = useVulkan && !vulkanUnavailable
                Log.i(
                    TAG,
                    "Recognition started: id=$requestId backend=${if (effectiveVulkan) "vulkan" else "cpu"}",
                )
                var inputFile: File? = null
                try {
                    currentCoroutineContext().ensureActive()
                    activeRequestId = requestId
                    inputFile = copyInput(image, requestId)
                    OcrNative.ensureLoaded()
                    nativeLoaded = true
                    val result = OcrNative.recognize(
                        modelStore.modelFile.absolutePath,
                        modelStore.mmprojFile.absolutePath,
                        inputFile.absolutePath,
                        recommendedThreadCount(),
                        effectiveVulkan,
                    )
                    notifyProgress(requestId, 100)
                    notifyResult(requestId, result)
                    Log.i(TAG, "Recognition finished: id=$requestId")
                } catch (error: Throwable) {
                    if (currentCoroutineContext().isActive) {
                        Log.e(TAG, "Recognition failed: id=$requestId", error)
                        notifyError(
                            requestId,
                            ERROR_INFERENCE,
                            error.message ?: "OCR inference failed",
                        )
                    }
                } finally {
                    inputFile?.delete()
                    if (activeRequestId == requestId) {
                        activeRequestId = NO_REQUEST
                    }
                }
            }
            if (requests.putIfAbsent(requestId, job) != null) {
                job.cancel()
                image.close()
                notifyError(requestId, ERROR_INVALID_REQUEST, "Invalid request ID")
                return
            }
            job.invokeOnCompletion {
                runCatching { image.close() }
                requests.remove(requestId, job)
                if (!shuttingDown) {
                    scheduleUnload(MODEL_IDLE_TIMEOUT_MS)
                }
            }
            job.start()
        }

        override fun cancelRecognition(requestId: Long) {
            val job = requests[requestId] ?: return
            if (activeRequestId == requestId && nativeLoaded) {
                runCatching { OcrNative.requestCancel() }
                    .onFailure { error -> Log.w(TAG, "Could not cancel OCR", error) }
            }
            job.cancel()
            Log.i(TAG, "Recognition cancelled: id=$requestId")
        }
    }

    override fun onCreate() {
        super.onCreate()
        modelStore = OcrModelStore(this, ::notifyModelState, ::notifyModelError)
        modelStore.resume()
    }

    override fun onBind(intent: Intent): IBinder? {
        if (intent.action != BIND_ACTION) return null
        if (intent.getBooleanExtra(EXTRA_PREWARM_OCR, false)) {
            prewarmModel(intent.getBooleanExtra(EXTRA_USE_VULKAN, false))
        }
        return binder
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (level >= TRIM_MEMORY_RUNNING_LOW && requests.isEmpty()) {
            scheduleUnload(0)
        }
    }

    override fun onDestroy() {
        shuttingDown = true
        cancelAllRecognition()
        modelStore.close()
        mainHandler.removeCallbacks(unloadRunnable)
        serviceScope.cancel()
        runBlocking(nativeDispatcher) {
            if (nativeLoaded) {
                runCatching { OcrNative.unload() }
                    .onFailure { error -> Log.w(TAG, "Could not release OCR model", error) }
                nativeLoaded = false
            }
        }
        nativeDispatcher.close()
        callbacks.kill()
        super.onDestroy()
    }

    private fun cancelAllRecognition() {
        if (activeRequestId != NO_REQUEST && nativeLoaded) {
            runCatching { OcrNative.requestCancel() }
                .onFailure { error -> Log.w(TAG, "Could not cancel OCR", error) }
        }
        requests.values.forEach(Job::cancel)
    }

    private fun prewarmModel(useVulkan: Boolean) {
        if (
            modelStore.snapshot.status != ModelStatus.READY ||
            nativeLoaded ||
            prewarmJob?.isActive == true
        ) {
            return
        }
        mainHandler.removeCallbacks(unloadRunnable)
        val job = serviceScope.launch(nativeDispatcher) {
            try {
                OcrNative.ensureLoaded()
                val loadedWithVulkan = OcrNative.prewarm(
                    modelStore.modelFile.absolutePath,
                    modelStore.mmprojFile.absolutePath,
                    recommendedThreadCount(),
                    useVulkan,
                )
                if (useVulkan && !loadedWithVulkan) {
                    vulkanUnavailable = true
                }
                nativeLoaded = true
                Log.i(TAG, "OCR model prewarmed with ${if (loadedWithVulkan) "Vulkan" else "CPU"}")
            } catch (error: Throwable) {
                Log.w(TAG, "Could not prewarm OCR model", error)
            }
        }
        prewarmJob = job
        job.invokeOnCompletion {
            if (prewarmJob === job) {
                prewarmJob = null
            }
            if (!shuttingDown) {
                scheduleUnload(MODEL_IDLE_TIMEOUT_MS)
            }
        }
    }

    private fun scheduleUnload(delayMillis: Long) {
        mainHandler.removeCallbacks(unloadRunnable)
        mainHandler.postDelayed(unloadRunnable, delayMillis)
    }

    private fun notifyModelState(snapshot: ModelSnapshot) {
        broadcast { callback ->
            callback.onModelStateChanged(
                snapshot.status,
                snapshot.downloadedBytes,
                snapshot.totalBytes,
            )
        }
    }

    private fun notifyModelError(message: String) {
        notifyError(NO_REQUEST, ERROR_DOWNLOAD, message)
    }

    private fun notifyProgress(requestId: Long, progress: Int) {
        broadcast { callback ->
            callback.onRecognitionProgress(requestId, progress)
        }
    }

    private fun notifyResult(requestId: Long, result: String) {
        broadcast { callback ->
            callback.onRecognitionResult(requestId, result)
        }
    }

    private fun notifyError(requestId: Long, errorCode: Int, message: String) {
        broadcast { callback ->
            callback.onError(requestId, errorCode, message)
        }
    }

    private fun broadcast(action: (IOcrCallback) -> Unit) {
        mainHandler.post {
            val count = callbacks.beginBroadcast()
            try {
                repeat(count) { index ->
                    runCatching { action(callbacks.getBroadcastItem(index)) }
                }
            } finally {
                callbacks.finishBroadcast()
            }
        }
    }

    private fun recommendedThreadCount(): Int {
        return Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
    }

    private fun copyInput(image: ParcelFileDescriptor, requestId: Long): File {
        val file = File(cacheDir, "ocr-input-$requestId-${System.nanoTime()}.png")
        try {
            ParcelFileDescriptor.AutoCloseInputStream(image).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
            return file
        } catch (error: Throwable) {
            file.delete()
            throw error
        }
    }

    private companion object {
        const val TAG = "uwuOcrService"
        const val BIND_ACTION = "org.uwuaosp.aicore.action.BIND_OCR"
        const val EXTRA_PREWARM_OCR = "org.uwuaosp.prism.extra.PREWARM_OCR"
        const val EXTRA_USE_VULKAN = "org.uwuaosp.prism.extra.USE_VULKAN"
        const val MODEL_IDLE_TIMEOUT_MS = 120_000L
        const val NO_REQUEST = -1L
        const val ERROR_MODEL_MISSING = 1
        const val ERROR_DOWNLOAD = 2
        const val ERROR_INFERENCE = 3
        const val ERROR_INVALID_REQUEST = 4
    }
}

private object ModelStatus {
    const val MISSING = 0
    const val DOWNLOADING = 1
    const val VERIFYING = 2
    const val READY = 3
    const val ERROR = 4
}

private data class ModelSnapshot(
    val status: Int,
    val downloadedBytes: Long,
    val totalBytes: Long,
)

private data class ModelAsset(
    val fileName: String,
    val url: String,
    val size: Long,
    val sha256: String,
)

private class OcrModelStore(
    private val context: Context,
    private val onStateChanged: (ModelSnapshot) -> Unit,
    private val onError: (String) -> Unit,
) {
    private val downloadManager = context.getSystemService(DownloadManager::class.java)
    private val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
    private val handler = Handler(Looper.getMainLooper())
    private val verifierDispatcher =
        Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val verifierScope = CoroutineScope(SupervisorJob())
    private var verificationJob: Job? = null
    private var verificationGeneration = 0L

    private val modelDirectory = File(
        requireNotNull(context.getExternalFilesDir(null)),
        MODEL_DIRECTORY,
    ).apply { mkdirs() }

    val modelFile = File(modelDirectory, MODEL_ASSET.fileName)
    val mmprojFile = File(modelDirectory, MMPROJ_ASSET.fileName)

    @Volatile
    var snapshot = initialSnapshot()
        private set

    private val pollRunnable = object : Runnable {
        override fun run() {
            runCatching { pollDownloads() }
                .onFailure { error ->
                    failDownload(error.message ?: "Could not query model download")
                }
            if (snapshot.status == ModelStatus.DOWNLOADING) {
                handler.postDelayed(this, POLL_INTERVAL_MS)
            }
        }
    }

    fun resume() {
        publish(snapshot)
        if (activeDownloadIds().isNotEmpty()) {
            updateSnapshot(ModelStatus.DOWNLOADING, 0)
            handler.post(pollRunnable)
        }
    }

    fun startDownload() {
        if (
            snapshot.status == ModelStatus.READY ||
            snapshot.status == ModelStatus.DOWNLOADING ||
            snapshot.status == ModelStatus.VERIFYING
        ) {
            return
        }
        cancelDownloadInternal(publishMissing = false)
        modelDirectory.mkdirs()

        var modelId = -1L
        var mmprojId = -1L
        try {
            modelId = enqueue(MODEL_ASSET)
            mmprojId = enqueue(MMPROJ_ASSET)
        } catch (error: Throwable) {
            val startedIds = listOf(modelId, mmprojId).filter { it >= 0 }
            if (startedIds.isNotEmpty()) {
                downloadManager.remove(*startedIds.toLongArray())
            }
            failDownload(error.message ?: "Could not start model download")
            return
        }
        preferences.edit()
            .putLong(PREF_MODEL_DOWNLOAD_ID, modelId)
            .putLong(PREF_MMPROJ_DOWNLOAD_ID, mmprojId)
            .putBoolean(PREF_VERIFIED, false)
            .apply()
        updateSnapshot(ModelStatus.DOWNLOADING, 0)
        handler.removeCallbacks(pollRunnable)
        handler.post(pollRunnable)
        Log.i(TAG, "Model download started")
    }

    fun cancelDownload() {
        if (
            snapshot.status != ModelStatus.DOWNLOADING &&
            snapshot.status != ModelStatus.VERIFYING
        ) {
            return
        }
        cancelDownloadInternal(publishMissing = true)
        Log.i(TAG, "Model download cancelled")
    }

    fun deleteModels() {
        cancelDownloadInternal(publishMissing = false)
        modelFile.delete()
        mmprojFile.delete()
        preferences.edit().putBoolean(PREF_VERIFIED, false).apply()
        updateSnapshot(ModelStatus.MISSING, 0)
        Log.i(TAG, "OCR models deleted")
    }

    fun close() {
        handler.removeCallbacks(pollRunnable)
        verificationJob?.cancel()
        verifierScope.cancel()
        verifierDispatcher.close()
    }

    private fun initialSnapshot(): ModelSnapshot {
        val verified = preferences.getBoolean(PREF_VERIFIED, false)
        return if (
            verified &&
            modelFile.length() == MODEL_ASSET.size &&
            mmprojFile.length() == MMPROJ_ASSET.size
        ) {
            ModelSnapshot(ModelStatus.READY, TOTAL_SIZE, TOTAL_SIZE)
        } else if (activeDownloadIds().isNotEmpty()) {
            ModelSnapshot(ModelStatus.DOWNLOADING, 0, TOTAL_SIZE)
        } else {
            ModelSnapshot(ModelStatus.MISSING, 0, TOTAL_SIZE)
        }
    }

    private fun enqueue(asset: ModelAsset): Long {
        partialFile(asset).delete()
        val request = DownloadManager.Request(Uri.parse(asset.url))
            .setTitle(asset.fileName)
            .setDescription("uwuPrism-OCR model")
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(false)
            .setNotificationVisibility(
                DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED,
            )
            .setDestinationInExternalFilesDir(
                context,
                null,
                "$MODEL_DIRECTORY/${asset.fileName}.download",
            )
        return downloadManager.enqueue(request)
    }

    private fun pollDownloads() {
        val ids = activeDownloadIds()
        if (ids.size != ASSETS.size) {
            failDownload("Missing DownloadManager request")
            return
        }

        var downloaded = 0L
        var completed = 0
        var found = 0
        val query = DownloadManager.Query().setFilterById(*ids.toLongArray())
        downloadManager.query(query)?.use { cursor ->
            val statusColumn = cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS)
            val downloadedColumn = cursor.getColumnIndexOrThrow(
                DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR,
            )
            val reasonColumn = cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON)
            while (cursor.moveToNext()) {
                found += 1
                downloaded += cursor.getLong(downloadedColumn).coerceAtLeast(0)
                when (cursor.getInt(statusColumn)) {
                    DownloadManager.STATUS_SUCCESSFUL -> completed += 1
                    DownloadManager.STATUS_FAILED -> {
                        failDownload(
                            "DownloadManager failed with reason ${cursor.getInt(reasonColumn)}",
                        )
                        return
                    }
                }
            }
        }
        if (found != ASSETS.size) {
            failDownload("DownloadManager request disappeared")
            return
        }

        updateSnapshot(ModelStatus.DOWNLOADING, downloaded)
        if (completed == ASSETS.size) {
            verifyDownloads()
        }
    }

    private fun verifyDownloads() {
        handler.removeCallbacks(pollRunnable)
        updateSnapshot(ModelStatus.VERIFYING, TOTAL_SIZE)
        verificationJob?.cancel()
        val generation = ++verificationGeneration
        verificationJob = verifierScope.launch(verifierDispatcher) {
            val error = runCatching {
                ASSETS.forEach { asset ->
                    currentCoroutineContext().ensureActive()
                    val partial = partialFile(asset)
                    check(partial.length() == asset.size) {
                        "${asset.fileName} has the wrong size"
                    }
                    check(sha256(partial) == asset.sha256) {
                        "${asset.fileName} failed SHA-256 verification"
                    }
                }
            }.exceptionOrNull()

            if (!currentCoroutineContext().isActive) return@launch
            handler.post {
                if (generation != verificationGeneration) return@post
                verificationJob = null
                if (error == null) {
                    val finalizeError = runCatching {
                        ASSETS.forEach { asset ->
                            val partial = partialFile(asset)
                            val final = File(modelDirectory, asset.fileName)
                            final.delete()
                            check(partial.renameTo(final)) {
                                "Could not finalize ${asset.fileName}"
                            }
                        }
                    }.exceptionOrNull()
                    if (finalizeError == null) {
                        clearDownloadIds()
                        preferences.edit().putBoolean(PREF_VERIFIED, true).apply()
                        updateSnapshot(ModelStatus.READY, TOTAL_SIZE)
                        Log.i(TAG, "OCR model verification completed")
                    } else {
                        failDownload(
                            finalizeError.message ?: "Could not finalize OCR model",
                        )
                    }
                } else {
                    failDownload(error.message ?: "Model verification failed")
                }
            }
        }
    }

    private fun failDownload(message: String) {
        cancelDownloadInternal(publishMissing = false)
        modelFile.delete()
        mmprojFile.delete()
        updateSnapshot(ModelStatus.ERROR, 0)
        onError(message)
        Log.e(TAG, "Model download failed: $message")
    }

    private fun cancelDownloadInternal(publishMissing: Boolean) {
        handler.removeCallbacks(pollRunnable)
        verificationGeneration += 1
        verificationJob?.cancel()
        verificationJob = null
        val ids = activeDownloadIds()
        if (ids.isNotEmpty()) {
            downloadManager.remove(*ids.toLongArray())
        }
        ASSETS.forEach { partialFile(it).delete() }
        clearDownloadIds()
        if (publishMissing) {
            updateSnapshot(ModelStatus.MISSING, 0)
        }
    }

    private fun activeDownloadIds(): List<Long> {
        return listOf(
            preferences.getLong(PREF_MODEL_DOWNLOAD_ID, -1),
            preferences.getLong(PREF_MMPROJ_DOWNLOAD_ID, -1),
        ).filter { it >= 0 }
    }

    private fun clearDownloadIds() {
        preferences.edit()
            .remove(PREF_MODEL_DOWNLOAD_ID)
            .remove(PREF_MMPROJ_DOWNLOAD_ID)
            .apply()
    }

    private fun partialFile(asset: ModelAsset): File {
        return File(modelDirectory, "${asset.fileName}.download")
    }

    private fun updateSnapshot(status: Int, downloadedBytes: Long) {
        publish(
            ModelSnapshot(
                status = status,
                downloadedBytes = downloadedBytes.coerceIn(0, TOTAL_SIZE),
                totalBytes = TOTAL_SIZE,
            ),
        )
    }

    private fun publish(value: ModelSnapshot) {
        snapshot = value
        onStateChanged(value)
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private companion object {
        const val TAG = "uwuOcrService"
        const val PREFERENCES = "ocr_model_store"
        const val PREF_MODEL_DOWNLOAD_ID = "model_download_id"
        const val PREF_MMPROJ_DOWNLOAD_ID = "mmproj_download_id"
        const val PREF_VERIFIED = "verified"
        const val MODEL_DIRECTORY = "ocr-models"
        const val POLL_INTERVAL_MS = 750L

        val MODEL_ASSET = ModelAsset(
            fileName = "glm-ocr-q4_k_m.gguf",
            url = "https://github.com/UwUniverse/GLM_OCR_GGUF/releases/download/v1.0.0/glm-ocr-q4_k_m.gguf",
            size = 548_510_048L,
            sha256 = "69c4ff4d7a87ad491bb69cc1f7b5c9d31a7ab53094b5bae5c299cdd1ffa81fe0",
        )
        val MMPROJ_ASSET = ModelAsset(
            fileName = "mmproj-GLM-OCR-Q8_0.gguf",
            url = "https://github.com/UwUniverse/GLM_OCR_GGUF/releases/download/v1.0.0/mmproj-GLM-OCR-Q8_0.gguf",
            size = 484_403_648L,
            sha256 = "9c4b58e33e316ed142eb5dcb41abec3844d3e6e5dc361ffb782c3fa9d175141f",
        )
        val ASSETS = listOf(MODEL_ASSET, MMPROJ_ASSET)
        val TOTAL_SIZE = ASSETS.sumOf(ModelAsset::size)
    }
}
