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
    // getJson,
    readResponseStream,
} from "./restClient.js";
import { TokenCounter } from "./tokenCounter.js";

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
        "http://localhost:8080",
    );
}

// ToDo: Implement model listing for BitNet if supported
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
    // BitNet server may not support model listing; fallback to env/config
    if (modelNames === undefined) {
        const modelName = getEnvSetting(env, "BITNET_MODEL", undefined, "bitnet_b1_58-3B");
        modelNames = [
            `bitnet:${modelName}`,
        ];
    }
    return modelNames;
}

export function bitnetApiSettingsFromEnv(
    modelType: ModelType,
    env: Record<string, string | undefined> = process.env,
    modelName: string = "bitnet_b1_58-3B",
): BitNetApiSettings {
    if (modelType === ModelType.Image) {
        throw new Error("Image model not supported");
    }
    const url = getBitNetEndpointUrl(env);
    return {
        provider: "bitnet",
        modelType,
        endpoint: `${url}/completion`,
        modelName,
    };
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
    model: string;
    created_at: string;
    message: {
        role: "assistant";
        content: string;
    };
    done: true;
} & BitNetChatCompletionUsage;

type BitNetChatCompletionChunk =
    | {
          model: string;
          created_at: string;
          done: false;
          message: {
              role: "assistant";
              content: string;
          };
      }
    | ({
          model: string;
          created_at: string;
          done: true;
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
        const messages =
            typeof prompt === "string"
                ? [{ role: "user", content: prompt }]
                : prompt;
        const isImageProptContent = (c: MultimodalPromptContent) =>
            (c as ImagePromptContent).type == "image_url";
        messages.map((ps) => {
            if (Array.isArray(ps.content)) {
                if (ps.content.some(isImageProptContent)) {
                    throw new Error("Image content not supported");
                }
            }
        });
        const params = {
            ...defaultParams,
            messages: messages,
            stream: false,
            options: completionSettings,
        };

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

        return success(data.message.content as string);
    }

    async function completeStream(
        prompt: string | PromptSection[],
    ): Promise<Result<AsyncIterableIterator<string>>> {
        const messages: PromptSection[] =
            typeof prompt === "string"
                ? [{ role: "user", content: prompt }]
                : prompt;

        const isImageProptContent = (c: MultimodalPromptContent) =>
            (c as ImagePromptContent).type == "image_url";
        messages.map((ps) => {
            if (Array.isArray(ps.content)) {
                if (ps.content.some(isImageProptContent)) {
                    throw new Error("Image content not supported");
                }
            }
        });

        const params = {
            ...defaultParams,
            messages: messages,
            stream: true,
            ...completionSettings,
        };
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
                const messageStream = readResponseStream(result.data);
                for await (const message of messageStream) {
                    const data: BitNetChatCompletionChunk = JSON.parse(message);
                    if (data.done) {
                        reportUsage(data);
                        break;
                    }
                    yield data.message.content;
                }
            })(),
        };
    }
}
