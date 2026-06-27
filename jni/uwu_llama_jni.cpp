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

#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "llama.h"
#include "llama-ext.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#define LOG_TAG "uwuLlama"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct ChatMessage {
    std::string role;
    std::string content;
};

std::atomic_bool g_backend_initialized{false};
std::atomic_bool g_stop_requested{false};

llama_model * g_model = nullptr;
llama_context * g_context = nullptr;
llama_sampler * g_sampler = nullptr;

std::vector<ChatMessage> g_messages;
std::string g_current_assistant;
std::string g_cached_token_bytes;
std::string g_last_error;
std::string g_recent_llama_log;
std::string g_status_log;
int g_previous_template_length = 0;
int g_remaining_tokens = 0;
bool g_generation_active = false;

void llama_log_callback(ggml_log_level level, const char * text, void *);

void append_status_line(const std::string & line) {
    if (line.empty()) {
        return;
    }
    if (!g_status_log.empty()) {
        g_status_log += "\n";
    }
    g_status_log += line;
    LOGI("%s", line.c_str());
}

void append_recent_llama_log(const char * text) {
    if (text == nullptr || text[0] == '\0') {
        return;
    }

    g_recent_llama_log += text;
    constexpr size_t max_log_size = 4096;
    if (g_recent_llama_log.size() > max_log_size) {
        g_recent_llama_log.erase(0, g_recent_llama_log.size() - max_log_size);
    }
}

std::string trim_copy(const std::string & value) {
    const char * whitespace = " \t\r\n";
    const size_t start = value.find_first_not_of(whitespace);
    if (start == std::string::npos) {
        return {};
    }
    const size_t end = value.find_last_not_of(whitespace);
    return value.substr(start, end - start + 1);
}

std::string error_with_recent_llama_log(const char * error) {
    const std::string recent_log = trim_copy(g_recent_llama_log);
    if (recent_log.empty()) {
        return error;
    }
    return std::string(error) + ": " + recent_log;
}

const char * backend_device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return "iGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return "accelerator";
        case GGML_BACKEND_DEVICE_TYPE_META:
            return "meta";
    }
    return "unknown";
}

std::string format_mib(size_t bytes) {
    std::ostringstream output;
    output << (bytes / 1024 / 1024) << " MiB";
    return output.str();
}

struct BackendDeviceStatus {
    int device_count = 0;
    int offload_device_count = 0;
};

BackendDeviceStatus append_backend_device_status() {
    BackendDeviceStatus status;
    const size_t count = ggml_backend_dev_count();
    status.device_count = static_cast<int>(count);
    append_status_line("Backend devices visible: " + std::to_string(status.device_count));

    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(dev, &props);

        const bool offload_device =
                props.type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                props.type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        if (offload_device) {
            ++status.offload_device_count;
        }

        std::string line = "Device " + std::to_string(i) + ": ";
        line += props.name != nullptr ? props.name : ggml_backend_dev_name(dev);
        line += " (";
        line += backend_device_type_name(props.type);
        line += ")";

        const char * description = props.description != nullptr
                ? props.description
                : ggml_backend_dev_description(dev);
        if (description != nullptr && description[0] != '\0') {
            line += " - ";
            line += description;
        }
        if (props.memory_total > 0) {
            line += ", memory ";
            line += format_mib(props.memory_free);
            line += " free / ";
            line += format_mib(props.memory_total);
            line += " total";
        }
        append_status_line(line);
    }

    append_status_line(std::string("GPU offload supported: ") +
            (status.offload_device_count > 0 ? "yes" : "no"));
    return status;
}

void append_model_device_status() {
    if (g_model == nullptr) {
        return;
    }

    const int device_count = llama_model_n_devices(g_model);
    append_status_line("Model loaded with " + std::to_string(device_count) + " device(s).");
    for (int i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = llama_model_get_device(g_model, i);
        if (dev == nullptr) {
            continue;
        }

        ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(dev, &props);
        std::string line = "Model device " + std::to_string(i) + ": ";
        line += props.name != nullptr ? props.name : ggml_backend_dev_name(dev);
        line += " (";
        line += backend_device_type_name(props.type);
        line += ")";
        append_status_line(line);
    }
}

void initialize_backend(bool allow_vulkan) {
    if (!g_backend_initialized.exchange(true)) {
        llama_log_set(llama_log_callback, nullptr);
        if (allow_vulkan) {
            unsetenv("GGML_DISABLE_VULKAN");
        } else {
            setenv("GGML_DISABLE_VULKAN", "1", 1);
        }
        llama_backend_init();
        unsetenv("GGML_DISABLE_VULKAN");
        LOGI("llama backend initialized");
    }

#ifdef GGML_USE_VULKAN
    if (allow_vulkan && ggml_backend_reg_by_name(GGML_VK_NAME) == nullptr) {
        ggml_backend_reg_t reg = ggml_backend_vk_reg();
        if (reg != nullptr) {
            ggml_backend_register(reg);
            append_status_line("Vulkan backend registered.");
        } else {
            append_status_line("Vulkan backend registration failed.");
        }
    }
#else
    if (allow_vulkan) {
        append_status_line("Vulkan backend is not compiled in.");
    }
#endif
}

void llama_log_callback(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        append_recent_llama_log(text);
        LOGE("%s", text);
    } else if (level >= GGML_LOG_LEVEL_WARN) {
        append_recent_llama_log(text);
        LOGW("%s", text);
    }
}

bool should_abort_decode(void *) {
    return g_stop_requested.load();
}

std::string to_string(JNIEnv * env, jstring value) {
    if (value == nullptr) {
        return {};
    }
    const char * chars = env->GetStringUTFChars(value, nullptr);
    std::string result(chars == nullptr ? "" : chars);
    if (chars != nullptr) {
        env->ReleaseStringUTFChars(value, chars);
    }
    return result;
}

jstring to_jstring(JNIEnv * env, const std::string & value) {
    return env->NewStringUTF(value.c_str());
}

jstring error_or_null(JNIEnv * env, const std::string & error) {
    if (error.empty()) {
        return nullptr;
    }
    g_last_error = error;
    return to_jstring(env, error);
}

bool is_valid_utf8(const std::string & value) {
    const unsigned char * bytes = reinterpret_cast<const unsigned char *>(value.c_str());
    while (*bytes != 0x00) {
        int count;
        if ((*bytes & 0x80) == 0x00) {
            count = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            count = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            count = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            count = 4;
        } else {
            return false;
        }

        ++bytes;
        for (int i = 1; i < count; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            ++bytes;
        }
    }
    return true;
}

void reset_chat_state() {
    g_messages.clear();
    g_current_assistant.clear();
    g_cached_token_bytes.clear();
    g_last_error.clear();
    g_previous_template_length = 0;
    g_remaining_tokens = 0;
    g_generation_active = false;
    g_stop_requested.store(false);
}

void free_model_state() {
    g_stop_requested.store(true);

    if (g_sampler != nullptr) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_context != nullptr) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    reset_chat_state();
}

std::string apply_llama_chat_template(
        const char * tmpl, const std::vector<llama_chat_message> & chat, bool add_assistant) {
    int32_t required = llama_chat_apply_template(
            tmpl, chat.data(), chat.size(), add_assistant, nullptr, 0);
    if (required <= 0) {
        return {};
    }

    std::vector<char> buffer(static_cast<size_t>(required) + 1);
    int32_t written = llama_chat_apply_template(
            tmpl, chat.data(), chat.size(), add_assistant, buffer.data(), buffer.size());
    if (written <= 0) {
        return {};
    }
    return std::string(buffer.data(), static_cast<size_t>(written));
}

std::string apply_gemma_chat_template(bool add_assistant) {
    std::string formatted;
    std::string system_prompt;
    for (const ChatMessage & message : g_messages) {
        std::string role = message.role;
        if (role == "system") {
            system_prompt += message.content;
            continue;
        }
        if (role == "assistant") {
            role = "model";
        }

        formatted += "<start_of_turn>";
        formatted += role;
        formatted += "\n";
        if (!system_prompt.empty() && role != "model") {
            formatted += system_prompt;
            formatted += "\n\n";
            system_prompt.clear();
        }
        formatted += message.content;
        formatted += "<end_of_turn>\n";
    }
    if (add_assistant) {
        formatted += "<start_of_turn>model\n";
    }
    return formatted;
}

std::string apply_chat_template(bool add_assistant) {
    if (g_model == nullptr) {
        return {};
    }

    std::vector<llama_chat_message> chat;
    chat.reserve(g_messages.size());
    for (const ChatMessage & message : g_messages) {
        chat.push_back({message.role.c_str(), message.content.c_str()});
    }

    const char * tmpl = llama_model_chat_template(g_model, nullptr);
    std::string formatted = apply_llama_chat_template(tmpl, chat, add_assistant);
    if (!formatted.empty()) {
        return formatted;
    }

    formatted = apply_llama_chat_template("gemma", chat, add_assistant);
    if (!formatted.empty()) {
        return formatted;
    }

    return apply_gemma_chat_template(add_assistant);
}

void finish_generation() {
    if (!g_generation_active) {
        return;
    }

    if (!g_current_assistant.empty()) {
        g_messages.push_back({"assistant", g_current_assistant});
        const std::string formatted = apply_chat_template(false);
        if (!formatted.empty()) {
            g_previous_template_length = static_cast<int>(formatted.size());
        }
    }

    g_current_assistant.clear();
    g_cached_token_bytes.clear();
    g_remaining_tokens = 0;
    g_generation_active = false;
}

llama_model_params make_model_params(bool use_vulkan, int gpu_layers) {
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = true;
    if (use_vulkan) {
        model_params.n_gpu_layers = gpu_layers;
    } else {
        model_params.n_gpu_layers = 0;
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.main_gpu = -1;
    }
    return model_params;
}

std::string load_model_once(
        const std::string & path,
        int context_size,
        int thread_count,
        const llama_model_params & model_params) {
    g_recent_llama_log.clear();

    LOGI("loading model from %s", path.c_str());
    g_model = llama_model_load_from_file(path.c_str(), model_params);
    if (g_model == nullptr) {
        return error_with_recent_llama_log("failed to load model");
    }

    const int threads = std::max(1, static_cast<int>(thread_count));
    const uint32_t ctx_size = static_cast<uint32_t>(std::max(1024, static_cast<int>(context_size)));
    const uint32_t batch_size = std::min<uint32_t>(512, ctx_size);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = ctx_size;
    ctx_params.n_batch = batch_size;
    ctx_params.n_ubatch = batch_size;
    ctx_params.n_threads = threads;
    ctx_params.n_threads_batch = threads;
    ctx_params.abort_callback = should_abort_decode;

    g_context = llama_init_from_model(g_model, ctx_params);
    if (g_context == nullptr) {
        return error_with_recent_llama_log("failed to create llama context");
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    reset_chat_state();
    LOGI("model loaded");
    return {};
}

std::string decode_prompt(const std::string & prompt) {
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(g_context), 0) == -1;
    const int32_t token_count = -llama_tokenize(
            vocab, prompt.c_str(), prompt.size(), nullptr, 0, is_first, true);
    if (token_count <= 0) {
        return "failed to tokenize prompt";
    }

    std::vector<llama_token> tokens(static_cast<size_t>(token_count));
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(),
            is_first, true) < 0) {
        return "failed to tokenize prompt";
    }

    const int used = llama_memory_seq_pos_max(llama_get_memory(g_context), 0) + 1;
    if (used + token_count > static_cast<int>(llama_n_ctx(g_context))) {
        return "context size exceeded";
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(g_context, batch) != 0) {
        return "llama_decode failed while processing prompt";
    }
    return {};
}

std::string decode_token(llama_token token) {
    llama_batch batch = llama_batch_get_one(&token, 1);
    if (llama_decode(g_context, batch) != 0) {
        return "llama_decode failed while generating token";
    }
    return {};
}

} // namespace

extern "C" JNIEXPORT void JNICALL
Java_org_uwuaosp_aicore_LlamaNative_initBackend(JNIEnv *, jobject) {
    initialize_backend(true);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_uwuaosp_aicore_LlamaNative_loadModel(
        JNIEnv * env,
        jobject,
        jstring model_path,
        jint context_size,
        jint thread_count,
        jboolean use_vulkan,
        jint gpu_layers) {
    free_model_state();
    g_stop_requested.store(false);
    g_status_log.clear();

    const std::string path = to_string(env, model_path);
    if (path.empty()) {
        return error_or_null(env, "empty model path");
    }

    const bool request_vulkan = use_vulkan == JNI_TRUE;
    const int requested_gpu_layers = static_cast<int>(gpu_layers);
    bool should_load_cpu = !request_vulkan;
    initialize_backend(request_vulkan);

    if (request_vulkan) {
        append_status_line("Vulkan requested; gpu layers: " +
                std::to_string(requested_gpu_layers));
        const BackendDeviceStatus device_status = append_backend_device_status();
        if (device_status.offload_device_count > 0) {
            const llama_model_params model_params =
                    make_model_params(true, requested_gpu_layers);
            const std::string error = load_model_once(
                    path, context_size, thread_count, model_params);
            if (error.empty()) {
                append_model_device_status();
                return nullptr;
            }

            append_status_line("Vulkan load failed: " + error);
            append_status_line("Fallback to CPU.");
            free_model_state();
            g_stop_requested.store(false);
            should_load_cpu = true;
        } else {
            append_status_line("Fallback to CPU: no GPU offload device visible.");
            should_load_cpu = true;
        }
    } else {
        append_status_line("Vulkan disabled; CPU backend selected.");
    }

    if (should_load_cpu) {
        const llama_model_params model_params = make_model_params(false, 0);
        const std::string error = load_model_once(
                path, context_size, thread_count, model_params);
        if (!error.empty()) {
            append_status_line("CPU load failed: " + error);
            free_model_state();
            return error_or_null(env, error);
        }
        append_model_device_status();
        return nullptr;
    }

    return error_or_null(env, "failed to choose llama backend");
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_uwuaosp_aicore_LlamaNative_beginPrompt(
        JNIEnv * env, jobject, jstring prompt, jint max_tokens) {
    g_last_error.clear();

    if (g_model == nullptr || g_context == nullptr || g_sampler == nullptr) {
        return error_or_null(env, "model is not loaded");
    }
    if (g_generation_active) {
        return error_or_null(env, "generation is already active");
    }

    const std::string user_prompt = to_string(env, prompt);
    if (user_prompt.empty()) {
        return error_or_null(env, "empty prompt");
    }

    g_stop_requested.store(false);
    g_current_assistant.clear();
    g_cached_token_bytes.clear();
    g_messages.push_back({"user", user_prompt});

    const std::string formatted = apply_chat_template(true);
    if (formatted.empty()) {
        g_messages.pop_back();
        return error_or_null(env, "failed to apply chat template");
    }

    const size_t start = std::min<size_t>(g_previous_template_length, formatted.size());
    const std::string prompt_delta = formatted.substr(start);
    const std::string decode_error = decode_prompt(prompt_delta);
    if (!decode_error.empty()) {
        g_messages.pop_back();
        return error_or_null(env, decode_error);
    }

    g_remaining_tokens = std::max(1, static_cast<int>(max_tokens));
    g_generation_active = true;
    return nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_uwuaosp_aicore_LlamaNative_nextToken(JNIEnv * env, jobject) {
    if (!g_generation_active || g_context == nullptr || g_model == nullptr || g_sampler == nullptr) {
        return nullptr;
    }
    if (g_stop_requested.load() || g_remaining_tokens <= 0) {
        finish_generation();
        return nullptr;
    }

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    const llama_token token = llama_sampler_sample(g_sampler, g_context, -1);
    if (llama_vocab_is_eog(vocab, token)) {
        finish_generation();
        return nullptr;
    }

    char piece[256];
    const int piece_size = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
    if (piece_size < 0) {
        g_last_error = "failed to convert token to text";
        finish_generation();
        return nullptr;
    }

    const std::string decode_error = decode_token(token);
    if (!decode_error.empty()) {
        g_last_error = decode_error;
        finish_generation();
        return nullptr;
    }

    --g_remaining_tokens;

    g_cached_token_bytes.append(piece, static_cast<size_t>(piece_size));
    if (!is_valid_utf8(g_cached_token_bytes)) {
        return to_jstring(env, "");
    }

    const std::string token_text = g_cached_token_bytes;
    g_current_assistant += token_text;
    g_cached_token_bytes.clear();

    if (g_remaining_tokens <= 0) {
        finish_generation();
    }
    return to_jstring(env, token_text);
}

extern "C" JNIEXPORT void JNICALL
Java_org_uwuaosp_aicore_LlamaNative_requestStop(JNIEnv *, jobject) {
    g_stop_requested.store(true);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_uwuaosp_aicore_LlamaNative_consumeStatusLog(JNIEnv * env, jobject) {
    if (g_status_log.empty()) {
        return nullptr;
    }
    const std::string status_log = g_status_log;
    g_status_log.clear();
    return to_jstring(env, status_log);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_uwuaosp_aicore_LlamaNative_consumeLastError(JNIEnv * env, jobject) {
    if (g_last_error.empty()) {
        return nullptr;
    }
    const std::string error = g_last_error;
    g_last_error.clear();
    return to_jstring(env, error);
}

extern "C" JNIEXPORT void JNICALL
Java_org_uwuaosp_aicore_LlamaNative_unload(JNIEnv *, jobject) {
    free_model_state();
}
