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

package org.uwuaosp.aicore

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.background
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.android.settingslib.collapsingtoolbar.CollapsingToolbarBaseActivity
import java.io.File
import java.io.IOException
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.uwuaosp.compose.settingslib.rememberSettingsTypography

class AiSettingsActivity : CollapsingToolbarBaseActivity() {
    private val activityScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private lateinit var controller: AiInferenceController

    private val modelPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        try {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION,
            )
        } catch (_: SecurityException) {
        }
        controller.selectModel(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setTitle(R.string.ai_settings_title)

        controller = AiInferenceController(this, activityScope)
        setContent {
            AiSettingsTheme {
                val state by controller.state.collectAsState()
                AiSettingsScreen(
                    state = state,
                    onActionClick = {
                        when (state.action) {
                            AiAction.SelectModel -> modelPicker.launch(arrayOf("*/*"))
                            AiAction.RunModel -> controller.loadSelectedModel()
                            AiAction.StopModel -> controller.stopModel()
                        }
                    },
                    onVulkanChange = controller::setUseVulkan,
                    onMessageChange = controller::updateMessage,
                    onSendClick = controller::sendMessage,
                )
            }
        }
    }

    override fun onDestroy() {
        controller.destroy()
        activityScope.cancel()
        super.onDestroy()
    }
}

private object LlamaNative {
    init {
        System.loadLibrary("uwu_aicore_llama_jni")
    }

    external fun initBackend()
    external fun loadModel(
        modelPath: String,
        contextSize: Int,
        threadCount: Int,
        useVulkan: Boolean,
        gpuLayers: Int,
    ): String?
    external fun beginPrompt(prompt: String, maxTokens: Int): String?
    external fun nextToken(): String?
    external fun requestStop()
    external fun consumeStatusLog(): String?
    external fun consumeLastError(): String?
    external fun isModelLoaded(): Boolean
    external fun unload()
}

private enum class AiRunStatus {
    NoModel,
    Ready,
    Loading,
    Running,
    Generating,
    Error,
}

private enum class AiAction {
    SelectModel,
    RunModel,
    StopModel,
}

private data class AiUiState(
    val selectedModelName: String? = null,
    val status: AiRunStatus = AiRunStatus.NoModel,
    val logText: String,
    val message: String = "",
    val useVulkan: Boolean = true,
) {
    val action: AiAction
        get() = when {
            selectedModelName == null -> AiAction.SelectModel
            status == AiRunStatus.Running || status == AiRunStatus.Generating -> AiAction.StopModel
            else -> AiAction.RunModel
        }

    val actionEnabled: Boolean
        get() = status != AiRunStatus.Loading

    val canSend: Boolean
        get() = status == AiRunStatus.Running && message.isNotBlank()

    val canChangeBackend: Boolean
        get() = status != AiRunStatus.Loading &&
            status != AiRunStatus.Running &&
            status != AiRunStatus.Generating
}

private class AiInferenceController(
    private val context: Context,
    private val scope: CoroutineScope,
) {
    private val nativeDispatcher: ExecutorCoroutineDispatcher =
        Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val logFileDispatcher: ExecutorCoroutineDispatcher =
        Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val logFile = File(context.filesDir, LOG_FILE_NAME)
    private val initialLog = context.getString(R.string.ai_log_initial)
    private val _state = MutableStateFlow(
        AiUiState(logText = initialLog),
    )
    val state: StateFlow<AiUiState> = _state.asStateFlow()

    private var selectedModelUri: Uri? = null
    private var generationJob: Job? = null
    private var operationSerial = 0

    init {
        scope.launch(logFileDispatcher) {
            runCatching {
                logFile.writeText(initialLog)
            }
        }
    }

    fun selectModel(uri: Uri) {
        stopModel(appendStoppedLog = false)
        if (!isGguf(uri)) {
            selectedModelUri = null
            appendLine(context.getString(R.string.ai_log_invalid_model))
            _state.update {
                it.copy(selectedModelName = null, status = AiRunStatus.NoModel)
            }
            return
        }

        val name = queryDisplayName(uri)
        selectedModelUri = uri
        _state.update {
            it.copy(
                selectedModelName = name,
                status = AiRunStatus.Ready,
                message = "",
            )
        }
        appendLine(context.getString(R.string.ai_log_model_selected, name))
    }

    fun updateMessage(message: String) {
        _state.update { it.copy(message = message) }
    }

    fun setUseVulkan(useVulkan: Boolean) {
        if (!_state.value.canChangeBackend) {
            return
        }
        _state.update { it.copy(useVulkan = useVulkan) }
    }

    fun loadSelectedModel() {
        val uri = selectedModelUri ?: return
        val name = _state.value.selectedModelName ?: queryDisplayName(uri)
        val useVulkan = _state.value.useVulkan
        if (_state.value.status == AiRunStatus.Loading ||
            _state.value.status == AiRunStatus.Running ||
            _state.value.status == AiRunStatus.Generating
        ) {
            return
        }

        val serial = nextOperationSerial()
        generationJob?.cancel()
        generationJob = scope.launch {
            _state.update { it.copy(status = AiRunStatus.Loading) }
            appendLine(context.getString(R.string.ai_log_model_preparing, name))
            val modelFileResult = withContext(Dispatchers.IO) {
                runCatching {
                    prepareLocalModelFile(uri, name)
                }
            }
            val modelFile = modelFileResult.getOrElse { error ->
                appendError("failed to prepare local model file: ${error.message ?: error.javaClass.simpleName}")
                if (serial == operationSerial) {
                    _state.update { it.copy(status = AiRunStatus.Error) }
                }
                return@launch
            }

            appendLine(context.getString(R.string.ai_log_model_loading, name))
            appendLine(
                context.getString(
                    if (useVulkan) {
                        R.string.ai_log_vulkan_requested
                    } else {
                        R.string.ai_log_vulkan_disabled
                    },
                ),
            )
            val (error, statusLog) = withContext(nativeDispatcher) {
                val loadError = LlamaNative.loadModel(
                    modelFile.absolutePath,
                    DEFAULT_CONTEXT_SIZE,
                    recommendedThreadCount(),
                    useVulkan,
                    DEFAULT_GPU_LAYERS,
                )
                loadError to LlamaNative.consumeStatusLog()
            }
            appendNativeStatusLog(statusLog)
            if (error != null) {
                appendError(error)
                if (serial == operationSerial) {
                    _state.update { it.copy(status = AiRunStatus.Error) }
                }
                return@launch
            }

            if (serial == operationSerial) {
                _state.update { it.copy(status = AiRunStatus.Running) }
                appendLine(context.getString(R.string.ai_log_model_ready))
            } else {
                withContext(nativeDispatcher) {
                    LlamaNative.unload()
                }
            }
        }
    }

    fun sendMessage() {
        val prompt = _state.value.message.trim()
        if (prompt.isEmpty() || _state.value.status != AiRunStatus.Running) {
            return
        }

        val serial = operationSerial
        _state.update { it.copy(message = "", status = AiRunStatus.Generating) }
        appendLine("${context.getString(R.string.ai_log_user_prefix)}: $prompt")

        generationJob = scope.launch {
            val (beginError, statusLog, modelLoaded) = withContext(nativeDispatcher) {
                val error = LlamaNative.beginPrompt(prompt, DEFAULT_MAX_TOKENS)
                Triple(
                    error,
                    LlamaNative.consumeStatusLog(),
                    error == null || LlamaNative.isModelLoaded(),
                )
            }
            appendNativeStatusLog(statusLog)
            if (beginError != null) {
                appendError(beginError)
                if (serial == operationSerial) {
                    _state.update {
                        it.copy(
                            status = if (modelLoaded) AiRunStatus.Running else AiRunStatus.Error,
                        )
                    }
                }
                return@launch
            }

            appendRaw("${context.getString(R.string.ai_log_assistant_prefix)}: ")
            withContext(nativeDispatcher) {
                while (isActive) {
                    val token = LlamaNative.nextToken() ?: break
                    if (token.isNotEmpty()) {
                        appendRaw(token)
                    }
                }
            }

            val nativeError = withContext(nativeDispatcher) {
                LlamaNative.consumeLastError()
            }
            appendRaw("\n")
            if (nativeError != null) {
                appendError(nativeError)
            }
            if (serial == operationSerial) {
                _state.update { it.copy(status = AiRunStatus.Running) }
            }
        }
    }

    fun stopModel(appendStoppedLog: Boolean = true) {
        val hadModel = _state.value.status == AiRunStatus.Loading ||
            _state.value.status == AiRunStatus.Running ||
            _state.value.status == AiRunStatus.Generating
        nextOperationSerial()
        generationJob?.cancel()
        LlamaNative.requestStop()
        scope.launch {
            withContext(nativeDispatcher) {
                LlamaNative.unload()
            }
            val nextStatus = if (selectedModelUri == null) AiRunStatus.NoModel else AiRunStatus.Ready
            _state.update { it.copy(status = nextStatus) }
            if (hadModel && appendStoppedLog) {
                appendLine(context.getString(R.string.ai_log_model_stopped))
            }
        }
    }

    fun destroy() {
        nextOperationSerial()
        generationJob?.cancel()
        LlamaNative.requestStop()
        runBlocking {
            withContext(nativeDispatcher) {
                LlamaNative.unload()
            }
        }
        nativeDispatcher.close()
        logFileDispatcher.close()
    }

    private fun nextOperationSerial(): Int {
        operationSerial += 1
        return operationSerial
    }

    private fun appendError(error: String) {
        appendLine(context.getString(R.string.ai_log_error, error))
    }

    private fun appendNativeStatusLog(statusLog: String?) {
        val log = statusLog?.trim().orEmpty()
        if (log.isNotEmpty()) {
            appendLine(log)
        }
    }

    private fun appendLine(line: String) {
        appendRaw("\n$line\n")
    }

    private fun appendRaw(text: String) {
        scope.launch(logFileDispatcher) {
            runCatching {
                logFile.appendText(text)
            }
        }
        _state.update { state ->
            val combined = (state.logText + text).takeLast(MAX_LOG_CHARS)
            state.copy(logText = combined)
        }
    }

    private fun queryDisplayName(uri: Uri): String {
        val cursor = context.contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            val index = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index >= 0 && it.moveToFirst()) {
                return it.getString(index)
            }
        }
        return uri.lastPathSegment ?: "model.gguf"
    }

    private fun prepareLocalModelFile(uri: Uri, displayName: String): File {
        val modelDir = File(context.filesDir, MODEL_DIR_NAME)
        if (!modelDir.exists() && !modelDir.mkdirs()) {
            throw IOException("failed to create ${modelDir.absolutePath}")
        }

        val modelName = sanitizeFileName(displayName)
        val target = File(modelDir, modelName)
        val expectedSize = queryFileSize(uri)
        if (target.isFile && expectedSize != null && target.length() == expectedSize) {
            return target
        }

        val temp = File(modelDir, "$modelName.tmp")
        context.contentResolver.openInputStream(uri)?.use { input ->
            temp.outputStream().use { output ->
                input.copyTo(output, bufferSize = MODEL_COPY_BUFFER_SIZE)
            }
        } ?: throw IOException("failed to open model input stream")

        if (expectedSize != null && temp.length() != expectedSize) {
            temp.delete()
            throw IOException("copied model size mismatch")
        }

        if (target.exists() && !target.delete()) {
            temp.delete()
            throw IOException("failed to replace existing local model")
        }
        if (!temp.renameTo(target)) {
            temp.copyTo(target, overwrite = true)
            temp.delete()
        }
        return target
    }

    private fun queryFileSize(uri: Uri): Long? {
        val cursor = context.contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            val index = it.getColumnIndex(OpenableColumns.SIZE)
            if (index >= 0 && it.moveToFirst() && !it.isNull(index)) {
                return it.getLong(index).takeIf { size -> size >= 0L }
            }
        }
        return null
    }

    private fun sanitizeFileName(name: String): String {
        val sanitized = name.replace(Regex("[^A-Za-z0-9._-]"), "_")
        return sanitized.ifBlank { "model.gguf" }
    }

    private fun isGguf(uri: Uri): Boolean {
        return try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                val magic = ByteArray(4)
                input.read(magic) == 4 &&
                    magic.contentEquals(byteArrayOf('G'.code.toByte(), 'G'.code.toByte(),
                        'U'.code.toByte(), 'F'.code.toByte()))
            } == true
        } catch (_: Exception) {
            false
        }
    }

    private fun recommendedThreadCount(): Int {
        return (Runtime.getRuntime().availableProcessors() - 2).coerceIn(2, 4)
    }

    companion object {
        private const val DEFAULT_CONTEXT_SIZE = 4096
        private const val DEFAULT_MAX_TOKENS = 1024
        private const val DEFAULT_GPU_LAYERS = -1
        private const val MAX_LOG_CHARS = 24000
        private const val LOG_FILE_NAME = "llama_run.log"
        private const val MODEL_DIR_NAME = "llama_models"
        private const val MODEL_COPY_BUFFER_SIZE = 1024 * 1024
    }
}

@Composable
private fun AiSettingsTheme(content: @Composable () -> Unit) {
    val context = LocalContext.current
    val darkTheme = isSystemInDarkTheme()
    val colorScheme = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        if (darkTheme) {
            dynamicDarkColorScheme(context)
        } else {
            dynamicLightColorScheme(context)
        }
    } else if (darkTheme) {
        darkColorScheme()
    } else {
        lightColorScheme()
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = rememberSettingsTypography(),
        content = content,
    )
}

@Composable
private fun AiSettingsScreen(
    state: AiUiState,
    onActionClick: () -> Unit,
    onVulkanChange: (Boolean) -> Unit,
    onMessageChange: (String) -> Unit,
    onSendClick: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .navigationBarsPadding()
            .padding(horizontal = 16.dp, vertical = 12.dp),
    ) {
        AiBackendControls(
            state = state,
            onVulkanChange = onVulkanChange,
        )
        Spacer(modifier = Modifier.height(10.dp))
        AiLogPanel(
            state = state,
            onActionClick = onActionClick,
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
        )
        Spacer(modifier = Modifier.height(10.dp))
        AiMessageBar(
            state = state,
            onMessageChange = onMessageChange,
            onSendClick = onSendClick,
        )
    }
}

@Composable
private fun AiBackendControls(
    state: AiUiState,
    onVulkanChange: (Boolean) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = stringResource(R.string.ai_vulkan_gpu),
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Switch(
            checked = state.useVulkan,
            onCheckedChange = onVulkanChange,
            enabled = state.canChangeBackend,
        )
    }
}

@Composable
private fun AiLogPanel(
    state: AiUiState,
    onActionClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val scrollState = rememberScrollState()
    LaunchedEffect(state.logText) {
        scrollState.animateScrollTo(scrollState.maxValue)
    }

    Box(modifier = modifier) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            shape = MaterialTheme.shapes.extraLarge,
            color = MaterialTheme.colorScheme.surfaceVariant,
            contentColor = MaterialTheme.colorScheme.onSurfaceVariant,
        ) {
            LogText(
                logText = state.logText,
                scrollState = scrollState,
                modifier = Modifier
                    .fillMaxSize()
                    .padding(14.dp),
            )
        }
        FloatingActionButton(
            onClick = {
                if (state.actionEnabled) {
                    onActionClick()
                }
            },
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(14.dp)
                .size(56.dp),
            shape = MaterialTheme.shapes.large,
            containerColor = MaterialTheme.colorScheme.primaryContainer,
            contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
        ) {
            Icon(
                imageVector = state.action.icon,
                contentDescription = stringResource(state.action.contentDescriptionRes),
            )
        }
    }
}

@Composable
private fun LogText(
    logText: String,
    scrollState: ScrollState,
    modifier: Modifier = Modifier,
) {
    SelectionContainer(
        modifier = modifier.verticalScroll(scrollState),
    ) {
        Text(
            text = logText,
            style = MaterialTheme.typography.bodyMedium,
            fontFamily = FontFamily.Monospace,
        )
    }
}

@Composable
private fun AiMessageBar(
    state: AiUiState,
    onMessageChange: (String) -> Unit,
    onSendClick: () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        OutlinedTextField(
            value = state.message,
            onValueChange = onMessageChange,
            modifier = Modifier
                .weight(1f)
                .heightIn(min = 56.dp, max = 112.dp),
            enabled = state.status == AiRunStatus.Running,
            shape = MaterialTheme.shapes.large,
            placeholder = {
                Text(
                    text = stringResource(R.string.ai_message_hint),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            },
            minLines = 1,
            maxLines = 3,
        )
        FilledIconButton(
            onClick = onSendClick,
            enabled = state.canSend,
            modifier = Modifier.size(56.dp),
            shape = MaterialTheme.shapes.large,
        ) {
            Icon(
                imageVector = Icons.Default.Send,
                contentDescription = stringResource(R.string.ai_action_send),
            )
        }
    }
}

private val AiAction.icon: ImageVector
    get() = when (this) {
        AiAction.SelectModel -> Icons.Default.Add
        AiAction.RunModel -> Icons.Default.PlayArrow
        AiAction.StopModel -> Icons.Default.Pause
    }

private val AiAction.contentDescriptionRes: Int
    get() = when (this) {
        AiAction.SelectModel -> R.string.ai_action_select_model
        AiAction.RunModel -> R.string.ai_action_run_model
        AiAction.StopModel -> R.string.ai_action_stop_model
    }
