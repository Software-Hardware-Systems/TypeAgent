/**
 * BitNet Usage and Performance Analysis
 *
 * Inference:
 *   - Use `run_inference.py` for single-prompt inference:
 *     python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "Your prompt here" -cnv
 *     - -m: Model path
 *     - -p: Prompt
 *     - -n: Number of tokens to predict
 *     - -t: Threads
 *     - -c: Context size
 *     - -temp: Temperature
 *     - -cnv: Chat mode (for instruct models)
 *
 * Server Mode:
 *   - Use `run_inference_server.py` to launch a local HTTP server for LLM inference.
 *   - The server is compatible with llama.cpp-style APIs and can be used as a local endpoint for TypeAgent.
 *
 * Benchmarking:
 *   - Use `utils/e2e_benchmark.py` for end-to-end performance benchmarking:
 *     python utils/e2e_benchmark.py -m /path/to/model -n 200 -p 256 -t 4
 *     - -m: Model path
 *     - -n: Number of tokens to generate
 *     - -p: Prompt length
 *     - -t: Threads
 *
 * Model Preparation:
 *   - Use `setup_env.py` and `generate-dummy-bitnet-model.py` for environment setup and dummy model generation.
 *
 * Performance:
 *   - BitNet achieves significant speedups and energy savings on both ARM and x86 CPUs, as detailed in the README and technical reports.
 *   - Quantization and kernel selection (I2_S, TL1, TL2) impact performance.
 *
 * Integration:
 *   - The BitNet server can be used as a drop-in local LLM endpoint for frameworks expecting a llama.cpp-compatible API.
 *   - For TypeAgent, configure the endpoint and model path in environment variables (e.g., BITNET_ENDPOINT, BITNET_MODEL).
 */
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import {
    ImagePromptContent,
    MultimodalPromptContent,
    PromptSection,
    Result,
    success,
} from "typechat";
import { getEnvSetting } from "./common.js";
import { ChatModelWithStreaming, CompletionSettings } from "./models.js";
import {
    CommonApiSettings,
    CompletionUsageStats,
    EnvVars,
    ModelType,
} from "./openai.js";
import {
    callApi,
    callJsonApi,
    getJson,
    readResponseStream,
} from "./restClient.js";
import { TokenCounter } from "./tokenCounter.js";
import { OpenAIApiSettings } from "./openaiSettings.js";

export type BitNetApiSettings = CommonApiSettings & {
    provider: "bitnet";
    modelType: ModelType;
    endpoint: string;
    modelName: string;
};

function getBitNetEndpointUrl(env: Record<string, string | undefined>) {
    return getEnvSetting(
        env,
        EnvVars.BITNET_ENDPOINT,
        undefined,
        "http://localhost:5005",
    );
}

// type BitNetTagResult = {
//     models: {
//         name: string;
//         modified_at: string;
//         size: number;
//         digest: string;
//         details: {
//             format: string;
//             family: string;
//             families: string[];
//             parameter_size: string;
//             quantization_level: string;
//         };
//     }[];
// };

let modelNames: string[] | undefined;
export async function getBitNetModelNames(
    env: Record<string, string | undefined> = process.env,
): Promise<string[]> {
    if (modelNames === undefined) {
        const url = `${getBitNetEndpointUrl(env).replace(/\/$/, "")}/props`;
        const result = await getJson({}, url, undefined);
        if (result.success) {
            // Extract model name from default_generation_settings.model
            const data = result.data as any;
            const modelPath = data?.default_generation_settings?.model;
            if (typeof modelPath === "string") {
                // Extract just the model directory name (e.g., "BitNet-b1.58-2B-4T")
                const match = modelPath.match(/models\/([^/]+)/);
                const modelName = match ? match[1] : modelPath;
                modelNames = [`bitnet:${modelName}`];
            } else {
                modelNames = [];
            }
        } else {
            modelNames = [];
        }
    }
    return modelNames;
}

export function bitnetApiSettingsFromEnv(
    modelType: ModelType,
    env: Record<string, string | undefined> = process.env,
    modelName: string = "bitnet_b1_58-3B",
): BitNetApiSettings | OpenAIApiSettings {
    const useOAIEndpoint = env["BITNET_USE_OAI_ENDPOINT"] !== "0";
    if (modelType === ModelType.Image) {
        throw new Error("Image model not supported");
    }
    const url = getBitNetEndpointUrl(env);
    if (useOAIEndpoint) {
        return {
            provider: "openai",
            modelType,
            endpoint:
                modelType === ModelType.Chat
                    ? `${url}/v1/chat/completions`
                    : `${url}/v1/embeddings`,
            modelName,
            apiKey: "",
            supportsResponseFormat: true, // REVIEW: just assume it supports it. Does BitNet reject this option?
        };
    } else {       
        return {
            provider: "bitnet",
            modelType,
            endpoint: `${url}/completion`,
            modelName,
        };
    }
}

type BitNetChatCompletionUsage = {
    total_duration: number;
    load_duration: number;
    prompt_eval_count: number;
    prompt_eval_duration: number;
    eval_count: number;
    eval_duration: number;
};


type BitNetChatCompletion = {
    content: string;
    stop: boolean;
    id_slot: number;
    multimodal: boolean;
    index: number;
} & BitNetChatCompletionUsage;

type BitNetChatCompletionChunk =
    | {
          content: string;
          stop: false;
          id_slot: number;
          multimodal: boolean;
          index: number;
      }
    | ({
          content: string;
          stop: true;
          id_slot: number;
          multimodal: boolean;
          index: number;
      } & BitNetChatCompletionUsage);

export function createBitNetChatModel(
    settings: BitNetApiSettings,
    completionSettings?: CompletionSettings,
    completionCallback?: (request: any, response: any) => void,
    tags?: string[],
) {
    completionSettings ??= {};
    completionSettings.n ??= 1;
    completionSettings.temperature ??= 0;

    const defaultParams = {
        model: settings.modelName,
    };
    const model: ChatModelWithStreaming = {
        completionSettings: completionSettings,
        completionCallback,
        complete,
        completeStream,
    };
    return model;

    function reportUsage(data: BitNetChatCompletionUsage) {
        try {
            // track token usage
            const usage: CompletionUsageStats = {
                completion_tokens: data.eval_count,
                prompt_tokens: data.prompt_eval_count,
                total_tokens: data.prompt_eval_count + data.eval_count,
            };

            TokenCounter.getInstance().add(usage, tags);
        } catch {}
    }

    async function complete(
        prompt: string | PromptSection[],
    ): Promise<Result<string>> {
        let params: any;
        if (settings.provider === "bitnet") {
            // Convert prompt to string for llama.cpp/BitNet
            let promptStr: string;
            if (typeof prompt === "string") {
                promptStr = prompt;
            } else {
                // Concatenate all prompt sections as plain text
                promptStr = prompt.map(ps => typeof ps.content === "string" ? ps.content : (Array.isArray(ps.content) ? ps.content.map(c => typeof c === "string" ? c : JSON.stringify(c)).join(" ") : "")).join("\n");
            }
            params = {
                ...defaultParams,
                prompt: promptStr,
                stream: false,
                options: completionSettings,
            };
        } else {
            // OpenAI-style
            const messages =
                typeof prompt === "string"
                    ? [{ role: "user", content: prompt }]
                    : prompt;
            const isImagePromptContent = (c: MultimodalPromptContent) =>
                (c as ImagePromptContent).type == "image_url";
            messages.map((ps) => {
                if (Array.isArray(ps.content)) {
                    if (ps.content.some(isImagePromptContent)) {
                        throw new Error("Image content not supported");
                    }
                }
            });
            params = {
                ...defaultParams,
                messages: messages,
                stream: false,
                options: completionSettings,
            };
        }

        const result = await callJsonApi(
            {},
            settings.endpoint,
            params,
            settings.maxRetryAttempts,
            settings.retryPauseMs,
            undefined,
            settings.throttler,
        );
        if (!result.success) {
            return result;
        }

        const data = result.data as BitNetChatCompletion;
        if (model.completionCallback) {
            model.completionCallback(params, data);
        }

        reportUsage(data);

        return success(data.content as string);
    }

    async function completeStream(
        prompt: string | PromptSection[],
    ): Promise<Result<AsyncIterableIterator<string>>> {
        /**
         * Handles SSE (Server-Sent Events) streaming responses from the BitNet server.
         * Each 'data:' line contains a JSON object matching BitNetChatCompletionChunk.
         * Yields data.content for each chunk, and stops on data.stop.
         */
        let params: any;
        if (settings.provider === "bitnet") {
            let promptStr: string;
            if (typeof prompt === "string") {
                promptStr = prompt;
            } else {
                promptStr = prompt.map(ps => typeof ps.content === "string" ? ps.content : (Array.isArray(ps.content) ? ps.content.map(c => typeof c === "string" ? c : JSON.stringify(c)).join(" ") : "")).join("\n");
            }
            params = {
                ...defaultParams,
                prompt: promptStr,
                stream: true,
                ...completionSettings,
            };
        } else {
            const messages: PromptSection[] =
                typeof prompt === "string"
                    ? [{ role: "user", content: prompt }]
                    : prompt;
            const isImagePromptContent = (c: MultimodalPromptContent) =>
                (c as ImagePromptContent).type == "image_url";
            messages.map((ps) => {
                if (Array.isArray(ps.content)) {
                    if (ps.content.some(isImagePromptContent)) {
                        throw new Error("Image content not supported");
                    }
                }
            });
            params = {
                ...defaultParams,
                messages: messages,
                stream: true,
                ...completionSettings,
            };
        }
        const result = await callApi(
            {},
            settings.endpoint,
            params,
            settings.maxRetryAttempts,
            settings.retryPauseMs,
        );
        if (!result.success) {
            return result;
        }
        return {
            success: true,
            data: (async function* () {
                let lastContentTimestamp = Date.now();
                const messageStream = readResponseStream(result.data);
                console.debug("[BitNet] Streaming response started", messageStream);
                
                // Track content state
                let contentAccumulated = "";
                let emptyMessageCount = 0;
                const MAX_EMPTY_MESSAGES = 5;
                const CONTENT_TIMEOUT_MS = 2000; // 2 seconds
                
                // Completion marker detection
                const COMPLETION_MARKERS = ["###", "```"];  // Common completion markers
                let pendingContent = "";
                
                for await (const message of messageStream) {
                    // Process each line in the message
                    const lines = message.split("\n");
                    let hasContent = false;
                    
                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed) continue;
                        if (!trimmed.startsWith("data:")) {
                            continue;
                        }
                        
                        const jsonStr = trimmed.slice(5).trim();
                        if (!jsonStr) continue;
                        
                        try {
                            const data: BitNetChatCompletionChunk = JSON.parse(jsonStr);
                            
                            // Check for explicit stop signal from server
                            if (data.stop === true) {
                                if (typeof reportUsage === "function" && 
                                    data.eval_count !== undefined && 
                                    data.prompt_eval_count !== undefined) {
                                    try {
                                        reportUsage(data as any);
                                    } catch (err) {
                                        console.error("[BitNet] Error reporting usage:", err);
                                    }
                                }
                                console.debug("[BitNet] Received explicit stop signal");
                                return;
                            }
                            
                            // Only yield if we have content and it's not a completion marker
                            if (typeof data.content === "string" && data.content.length > 0) {
                                // Check if this chunk contains any of our completion markers
                                const isCompletionMarker = COMPLETION_MARKERS.some(marker => {
                                    // Check full content with what's pending
                                    const fullContent = pendingContent + data.content;
                                    // Look for marker followed by newlines at the end
                                    return fullContent.endsWith(marker) || 
                                           fullContent.endsWith(marker + "\n") ||
                                           fullContent.endsWith(marker + "\n\n");
                                });
                                
                                if (isCompletionMarker) {
                                    console.debug("[BitNet] Detected completion marker, not yielding:", data.content);
                                    
                                    // If we've accumulated substantial content before this marker,
                                    // consider this the end of the response
                                    if (contentAccumulated.length > 10) {
                                        console.debug("[BitNet] Terminating stream after completion marker");
                                        return;
                                    }
                                } else {
                                    // Add to pending content to check for markers that span chunks
                                    pendingContent += data.content;
                                    if (pendingContent.length > 20) {
                                        pendingContent = pendingContent.slice(-20); // Keep last 20 chars
                                    }
                                    
                                    // Yield the content
                                    yield data.content;
                                    contentAccumulated += data.content;
                                    lastContentTimestamp = Date.now();
                                    hasContent = true;
                                    emptyMessageCount = 0; // Reset counter when we get content
                                }
                            }
                        } catch (err) {
                            console.debug("[BitNet] Ignored malformed JSON line:", jsonStr);
                        }
                    }
                    
                    // Count empty messages or check for logical completion
                    if (!hasContent) {
                        emptyMessageCount++;
                        
                        // Terminate if we've seen enough empty messages
                        if (emptyMessageCount >= MAX_EMPTY_MESSAGES) {
                            console.debug(`[BitNet] ${MAX_EMPTY_MESSAGES} consecutive empty messages, terminating stream`);
                            break;
                        }
                        
                        // Terminate if we haven't received content for a while and we have some accumulated content
                        if (contentAccumulated.length > 0 && 
                            Date.now() - lastContentTimestamp > CONTENT_TIMEOUT_MS) {
                            console.debug(`[BitNet] No new content for ${CONTENT_TIMEOUT_MS}ms with existing content, terminating stream`);
                            break;
                        }
                    }
                }
                
                console.debug("[BitNet] Stream complete");
            })(),
        };
    }
}
