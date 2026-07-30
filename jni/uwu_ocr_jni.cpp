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
#include <cstdlib>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#define LOG_TAG "uwuOcrNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::atomic_bool g_backend_initialized{false};
std::atomic_bool g_cancel_requested{false};

llama_model * g_model = nullptr;
mtmd_context * g_mtmd = nullptr;
std::string g_model_path;
std::string g_mmproj_path;
bool g_loaded_with_vulkan = false;
int g_loaded_threads = 0;

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

bool should_abort(void *) {
    return g_cancel_requested.load();
}

bool load_progress(float, void *) {
    return !g_cancel_requested.load();
}

void log_callback(ggml_log_level level, const char * text, void *) {
    if (text == nullptr) {
        return;
    }
    if (level >= GGML_LOG_LEVEL_ERROR) {
        LOGE("%s", text);
    } else if (level >= GGML_LOG_LEVEL_WARN) {
        LOGW("%s", text);
    }
}

void initialize_backend(bool allow_vulkan) {
    if (!g_backend_initialized.exchange(true)) {
        llama_log_set(log_callback, nullptr);
        mtmd_helper_log_set(log_callback, nullptr);
        if (allow_vulkan) {
            unsetenv("GGML_DISABLE_VULKAN");
        } else {
            setenv("GGML_DISABLE_VULKAN", "1", 1);
        }
        llama_backend_init();
        unsetenv("GGML_DISABLE_VULKAN");
    }

#ifdef GGML_USE_VULKAN
    if (allow_vulkan && ggml_backend_reg_by_name(GGML_VK_NAME) == nullptr) {
        ggml_backend_reg_t reg = ggml_backend_vk_reg();
        if (reg != nullptr) {
            ggml_backend_register(reg);
        }
    }
#endif
}

bool has_offload_device() {
    const size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_props properties = {};
        ggml_backend_dev_get_props(ggml_backend_dev_get(i), &properties);
        if (properties.type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                properties.type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return true;
        }
    }
    return false;
}

void free_loaded_model() {
    if (g_mtmd != nullptr) {
        mtmd_free(g_mtmd);
        g_mtmd = nullptr;
    }
    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_model_path.clear();
    g_mmproj_path.clear();
    g_loaded_with_vulkan = false;
    g_loaded_threads = 0;
}

llama_model_params make_model_params(bool use_vulkan) {
    llama_model_params params = llama_model_default_params();
    params.use_mmap = true;
    params.progress_callback = load_progress;
    if (use_vulkan) {
        params.n_gpu_layers = -1;
    } else {
        params.n_gpu_layers = 0;
        params.split_mode = LLAMA_SPLIT_MODE_NONE;
        params.main_gpu = -1;
    }
    return params;
}

std::string ensure_model_loaded(
        const std::string & model_path,
        const std::string & mmproj_path,
        int thread_count,
        bool use_vulkan) {
    if (g_model != nullptr &&
            g_mtmd != nullptr &&
            g_model_path == model_path &&
            g_mmproj_path == mmproj_path &&
            g_loaded_with_vulkan == use_vulkan &&
            g_loaded_threads == thread_count) {
        return {};
    }

    free_loaded_model();
    initialize_backend(use_vulkan);
    if (use_vulkan && !has_offload_device()) {
        return "no Vulkan offload device is available";
    }

    g_model = llama_model_load_from_file(
            model_path.c_str(),
            make_model_params(use_vulkan));
    if (g_model == nullptr) {
        free_loaded_model();
        return "failed to load GLM-OCR model";
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = use_vulkan;
    mtmd_params.print_timings = false;
    mtmd_params.n_threads = thread_count;
    mtmd_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    mtmd_params.warmup = false;
    mtmd_params.progress_callback = load_progress;
    g_mtmd = mtmd_init_from_file(mmproj_path.c_str(), g_model, mtmd_params);
    if (g_mtmd == nullptr || !mtmd_support_vision(g_mtmd)) {
        free_loaded_model();
        return "failed to load GLM-OCR multimodal projector";
    }

    g_model_path = model_path;
    g_mmproj_path = mmproj_path;
    g_loaded_with_vulkan = use_vulkan;
    g_loaded_threads = thread_count;
    LOGI("OCR model loaded with %s", use_vulkan ? "Vulkan" : "CPU");
    return {};
}

std::string apply_chat_template() {
    const std::string content = std::string(mtmd_get_marker(g_mtmd)) + "\nOCR";
    const llama_chat_message message = {
        /* .role = */ "user",
        /* .content = */ content.c_str(),
    };
    const char * tmpl = llama_model_chat_template(g_model, nullptr);
    int32_t required = llama_chat_apply_template(
            tmpl,
            &message,
            1,
            true,
            nullptr,
            0);
    if (required <= 0) {
        return {};
    }

    std::vector<char> output(static_cast<size_t>(required) + 1);
    const int32_t written = llama_chat_apply_template(
            tmpl,
            &message,
            1,
            true,
            output.data(),
            static_cast<int32_t>(output.size()));
    if (written <= 0) {
        return {};
    }
    return std::string(output.data(), static_cast<size_t>(written));
}

std::string token_to_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(128);
    int32_t length = llama_token_to_piece(
            vocab,
            token,
            buffer.data(),
            static_cast<int32_t>(buffer.size()),
            0,
            true);
    if (length < 0) {
        buffer.resize(static_cast<size_t>(-length));
        length = llama_token_to_piece(
                vocab,
                token,
                buffer.data(),
                static_cast<int32_t>(buffer.size()),
                0,
                true);
    }
    return length > 0 ? std::string(buffer.data(), static_cast<size_t>(length)) : std::string();
}

std::string recognize_once(
        const std::string & model_path,
        const std::string & mmproj_path,
        const std::string & image_path,
        int thread_count,
        bool use_vulkan,
        std::string * error) {
    *error = ensure_model_loaded(
            model_path,
            mmproj_path,
            thread_count,
            use_vulkan);
    if (!error->empty()) {
        return {};
    }

    mtmd_helper_bitmap_wrapper bitmap_wrapper =
            mtmd_helper_bitmap_init_from_file(g_mtmd, image_path.c_str(), false);
    if (bitmap_wrapper.bitmap == nullptr) {
        *error = "failed to decode OCR image";
        return {};
    }

    llama_context * context = nullptr;
    llama_sampler * sampler = nullptr;
    mtmd_input_chunks * chunks = nullptr;
    std::string result;

    do {
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 12288;
        context_params.n_batch = 512;
        context_params.n_ubatch = 512;
        context_params.n_threads = thread_count;
        context_params.n_threads_batch = thread_count;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_params.abort_callback = should_abort;
        context = llama_init_from_model(g_model, context_params);
        if (context == nullptr) {
            *error = "failed to create OCR context";
            break;
        }

        const std::string prompt = apply_chat_template();
        if (prompt.empty()) {
            *error = "failed to apply GLM-OCR chat template";
            break;
        }

        mtmd_input_text input = {
            /* .text = */ prompt.c_str(),
            /* .add_special = */ true,
            /* .parse_special = */ true,
        };
        chunks = mtmd_input_chunks_init();
        const mtmd_bitmap * bitmaps[] = {bitmap_wrapper.bitmap};
        if (mtmd_tokenize(g_mtmd, chunks, &input, bitmaps, 1) != 0) {
            *error = "failed to tokenize OCR input";
            break;
        }

        llama_pos n_past = 0;
        if (mtmd_helper_eval_chunks(
                    g_mtmd,
                    context,
                    chunks,
                    0,
                    0,
                    512,
                    true,
                    &n_past) != 0) {
            *error = g_cancel_requested.load()
                    ? "OCR request cancelled"
                    : "failed to encode OCR image";
            break;
        }

        sampler = llama_sampler_init_greedy();
        const llama_vocab * vocab = llama_model_get_vocab(g_model);
        for (int generated = 0; generated < 2048; ++generated) {
            if (g_cancel_requested.load()) {
                *error = "OCR request cancelled";
                break;
            }
            llama_token token = llama_sampler_sample(sampler, context, -1);
            llama_sampler_accept(sampler, token);
            if (llama_vocab_is_eog(vocab, token)) {
                break;
            }
            result += token_to_piece(vocab, token);

            llama_batch batch = llama_batch_get_one(&token, 1);
            if (llama_decode(context, batch) != 0) {
                *error = g_cancel_requested.load()
                        ? "OCR request cancelled"
                        : "failed to decode OCR output";
                break;
            }
        }
    } while (false);

    if (chunks != nullptr) {
        mtmd_input_chunks_free(chunks);
    }
    if (sampler != nullptr) {
        llama_sampler_free(sampler);
    }
    if (context != nullptr) {
        llama_free(context);
    }
    mtmd_bitmap_free(bitmap_wrapper.bitmap);
    if (bitmap_wrapper.video_ctx != nullptr) {
        mtmd_helper_video_free(bitmap_wrapper.video_ctx);
    }
    return result;
}

jstring throw_error(JNIEnv * env, const std::string & message) {
    jclass exception_class = env->FindClass("java/lang/IllegalStateException");
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message.c_str());
    }
    return nullptr;
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_org_uwuaosp_aicore_ocr_OcrNative_recognize(
        JNIEnv * env,
        jobject,
        jstring model_path,
        jstring mmproj_path,
        jint image_fd,
        jint thread_count,
        jboolean use_vulkan) {
    g_cancel_requested.store(false);
    const std::string model = to_string(env, model_path);
    const std::string mmproj = to_string(env, mmproj_path);
    if (model.empty() || mmproj.empty() || image_fd < 0) {
        return throw_error(env, "invalid OCR input");
    }

    const std::string image = "/proc/self/fd/" + std::to_string(image_fd);
    const int threads = std::max(1, static_cast<int>(thread_count));
    std::string error;
    std::string result = recognize_once(
            model,
            mmproj,
            image,
            threads,
            use_vulkan == JNI_TRUE,
            &error);

    if (!error.empty() && use_vulkan == JNI_TRUE && !g_cancel_requested.load()) {
        LOGW("Vulkan OCR failed; retrying on CPU: %s", error.c_str());
        free_loaded_model();
        error.clear();
        result = recognize_once(
                model,
                mmproj,
                image,
                threads,
                false,
                &error);
    }

    if (!error.empty()) {
        if (use_vulkan == JNI_TRUE) {
            free_loaded_model();
        }
        return throw_error(env, error);
    }
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_org_uwuaosp_aicore_ocr_OcrNative_requestCancel(JNIEnv *, jobject) {
    g_cancel_requested.store(true);
}

extern "C" JNIEXPORT void JNICALL
Java_org_uwuaosp_aicore_ocr_OcrNative_unload(JNIEnv *, jobject) {
    g_cancel_requested.store(true);
    free_loaded_model();
}
