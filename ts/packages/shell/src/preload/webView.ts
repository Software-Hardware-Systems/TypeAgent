// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

const { contextBridge } = require("electron/renderer");

import { ipcRenderer } from "electron";
import registerDebug from "debug";
import {
    setupInlineBrowserRendererProxy,
    sendScriptAction,
} from "./inlineBrowserRendererRpcServer";
import { ElectronPDFInterceptor } from "./pdfInterceptor";

const debugWebAgentProxy = registerDebug("typeagent:webAgent:proxy");

ipcRenderer.on("received-from-browser-ipc", async (_, data) => {
    if (data.error) {
        console.error(data.error);
        return;
    }

    if (data.method !== undefined && data.method.indexOf("/") > 0) {
        const [schema, actionName] = data.method?.split("/");

        if (schema === "browser") {
            const response = await runBrowserAction({
                actionName: actionName,
                parameters: data.params,
            });

            sendToBrowserAgent({
                id: data.id,
                result: response,
            });
        } else if (schema.startsWith("browser.")) {
            const message = await runSiteAction(schema, {
                actionName: actionName,
                parameters: data.params,
            });

            sendToBrowserAgent({
                id: data.id,
                result: message,
            });
        } else if (schema === "webAgent") {
            debugWebAgentProxy(`Dispatcher -> WebAgent`, data);
            window.postMessage(data);
        }

        console.log(
            `Browser websocket client received message: ${JSON.stringify(data)}`,
        );
    }
});

// Set up inline browser renderer RPC proxy
setupInlineBrowserRendererProxy();

// Initialize PDF interceptor when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
    const pdfInterceptor = new ElectronPDFInterceptor();
    pdfInterceptor.initialize();
});

function sendToBrowserAgent(message: any) {
    ipcRenderer.send("send-to-browser-ipc", message);
}

export async function awaitPageLoad() {
    /*
    return new Promise<string | undefined>((resolve, reject) => {
        // use window API to await pageload
        
    });
    */
}

export async function getTabHTMLFragments(
    fullSize: boolean,
    extractText: boolean,
) {
    let htmlFragments: any[] = [];
    let htmlPromises: Promise<any>[] = [];

    htmlPromises.push(
        sendScriptAction(
            {
                type: "get_reduced_html",
                fullSize: fullSize,
                frameId: 0,
            },
            50000,
            window.top,
            "0",
        ),
    );

    const iframeElements = document.getElementsByTagName("iframe");
    for (let i = 0; i < iframeElements.length; i++) {
        const frameElement = iframeElements[i];
        if (
            !frameElement.src ||
            frameElement.src == "about:blank" ||
            frameElement.hidden ||
            (frameElement.clientHeight == 0 && frameElement.clientWidth == 0)
        ) {
            continue;
        }

        const index = i + 1;
        htmlPromises.push(
            sendScriptAction(
                {
                    type: "get_reduced_html",
                    fullSize: fullSize,
                    frameId: index,
                },
                50000,
                frameElement.contentWindow,
                index.toString(),
            ),
        );
    }

    const htmlResults = await Promise.all(htmlPromises);
    for (let i = 0; i < htmlResults.length; i++) {
        const frameHTML = htmlResults[i];
        if (frameHTML) {
            let frameText = "";
            if (extractText) {
                frameText = await sendScriptAction(
                    {
                        type: "get_page_text",
                        inputHtml: frameHTML,
                        frameId: i,
                    },
                    1000,
                    frames[i],
                );
            }

            htmlFragments.push({
                frameId: i,
                content: frameHTML,
                text: frameText,
            });
        }
    }

    return htmlFragments;
}

export async function sendScriptActionToAllFrames(body: any, timeout?: number) {
    const frames = [window.top, ...Array.from(window.frames)];

    let htmlPromises: Promise<any>[] = [];
    frames.forEach((frame, index) => {
        htmlPromises.push(
            sendScriptAction(body, timeout, frame, index.toString()),
        );
    });

    return await Promise.all(htmlPromises);
}

async function runBrowserAction(action: any) {
    let responseObject: any;
    let confirmationMessage = "OK";
    const actionName =
        action.actionName ?? action.fullActionName.split(".").at(-1);
    switch (actionName) {
        case "getHTML": {
            responseObject = await getTabHTMLFragments(
                action.parameters.fullHTML,
                action.parameters?.extractText,
            );

            break;
        }

        case "clickOnElement":
        case "enterTextInElement":
        case "enterTextOnPage": {
            sendScriptActionToAllFrames({
                type: "run_ui_event",
                action: action,
            });
            break;
        }

        default:
            throw new Error(`Invalid action: ${actionName}`);
    }

    return {
        message: confirmationMessage,
        data: responseObject,
    };
}

async function runSiteAction(schemaName: string, action: any) {
    let confirmationMessage = "OK";
    switch (schemaName) {
        case "browser.crossword": {
            sendScriptAction({
                type: "run_crossword_action",
                action: action,
            });

            break;
        }
        case "browser.commerce": {
            sendScriptAction({
                type: "run_commerce_action",
                action: action,
            });

            break;
        }
    }

    return confirmationMessage;
}

ipcRenderer.invoke("init-browser-ipc");

contextBridge.exposeInMainWorld("browserConnect", {
    enableSiteAgent: (schemaName) => {
        if (schemaName) {
            sendToBrowserAgent({
                method: "enableSiteTranslator",
                params: { translator: schemaName },
            });
        }
    },
    disableSiteAgent: (schemaName) => {
        if (schemaName) {
            sendToBrowserAgent({
                method: "disableSiteTranslator",
                params: { translator: schemaName },
            });
        }
    },
});

// Add extension service adapter API for view pages
contextBridge.exposeInMainWorld("electronAPI", {
    // Extension service adapter API
    sendBrowserMessage: async (message: any) => {
        return ipcRenderer.invoke("browser-extension-message", message);
    },

    // Storage API for extension compatibility
    getStorage: async (keys: string[]) => {
        return ipcRenderer.invoke("extension-storage-get", keys);
    },

    setStorage: async (items: Record<string, any>) => {
        return ipcRenderer.invoke("extension-storage-set", items);
    },

    // Direct WebSocket connection check
    checkWebSocketConnection: async () => {
        return ipcRenderer.invoke("check-websocket-connection");
    },
});

window.addEventListener("message", async (event) => {
    if (event.data !== undefined && event.data.source === "webAgent") {
        debugWebAgentProxy(`WebAgent -> Dispatcher`, event.data);
        if (event.data.method === "webAgent/register") {
            // Fill in identification information
            event.data.params.param.title = document.title;
            event.data.params.param.url = window.location.href;
        }
        sendToBrowserAgent(event.data);
    }
});
