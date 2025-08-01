// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
import { build } from "vite";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { copyFileSync, mkdirSync, cpSync } from "fs";
import chalk from "chalk";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Helper function to create Vite build options that avoid TypeScript plugin conflicts
function createBuildOptions(outDir, options = {}) {
    return {
        configFile: false,
        logLevel: "error",
        plugins: [
            // Explicitly disable TypeScript plugin to avoid outDir conflicts
            {
                name: "disable-typescript-plugin",
                configResolved(config) {
                    // Remove TypeScript plugin
                    config.plugins = config.plugins.filter(
                        (plugin) =>
                            !plugin.name || !plugin.name.includes("typescript"),
                    );
                },
            },
        ],
        esbuild: {
            // Use esbuild for TypeScript compilation
            target: "es2022",
        },
        build: {
            outDir,
            emptyOutDir: options.emptyOutDir ?? false,
            sourcemap: isDev,
            minify: !isDev,
            rollupOptions: options.rollupOptions || {},
        },
    };
}

const isDev =
    process.argv.includes("--dev") ||
    process.argv.includes("--mode=development");
const buildMode = isDev ? "development" : "production";
const verbose = process.argv.includes("--verbose");

const chromeOutDir = resolve(__dirname, "../dist/extension");
const electronOutDir = resolve(__dirname, "../dist/electron");
const srcDir = resolve(__dirname, "../src/extension");
const electronSrcDir = resolve(__dirname, "../src/electron");

const sharedScripts = {
    contentScript: "contentScript/index.ts",
    webTypeAgentMain: "webTypeAgentMain.ts",
    webTypeAgentContentScript: "webTypeAgentContentScript.ts",
    "views/options": "views/options.ts",
    "views/pageMacros": "views/pageMacros.ts",
    "views/macrosLibrary": "views/macrosLibrary.ts",
    "views/entityGraphView": "views/entityGraphView.ts",
    "views/pageKnowledge": "views/pageKnowledge.ts",
    "views/annotationsLibrary": "views/annotationsLibrary.ts",
    "views/knowledgeLibrary": "views/knowledgeLibrary.ts",
    "views/pdfView": "views/pdfView.ts",
    uiEventsDispatcher: "uiEventsDispatcher.ts",
    "sites/paleobiodb": "sites/paleobiodb.ts",
    "offscreen/contentProcessor": "offscreen/contentProcessor.ts",
};

const electronOnlyScripts = {
    agentActivation: "../src/electron/agentActivation.ts",
};

const vendorAssets = [
    [
        "node_modules/bootstrap/dist/css/bootstrap.min.css",
        "vendor/bootstrap/bootstrap.min.css",
    ],
    [
        "node_modules/bootstrap/dist/js/bootstrap.bundle.min.js",
        "vendor/bootstrap/bootstrap.bundle.min.js",
    ],
    ["node_modules/prismjs/prism.js", "vendor/prism/prism.js"],
    ["node_modules/prismjs/themes/prism.css", "vendor/prism/prism.css"],
    [
        "node_modules/prismjs/components/prism-typescript.js",
        "vendor/prism/prism-typescript.js",
    ],
    [
        "node_modules/prismjs/components/prism-json.js",
        "vendor/prism/prism-json.js",
    ],
    // Cytoscape libraries for entity graph visualization
    [
        "node_modules/cytoscape/dist/cytoscape.min.js",
        "vendor/cytoscape/cytoscape.min.js",
    ],
    ["node_modules/dagre/dist/dagre.min.js", "vendor/dagre/dagre.min.js"],
    [
        "node_modules/cytoscape-dagre/cytoscape-dagre.js",
        "vendor/cytoscape-dagre/cytoscape-dagre.min.js",
    ],
];

const libraryAssets = [
    "views/annotationsLibrary.css",
    "views/annotationsLibrary.html",
    "views/entityGraphView.css",
    "views/entityGraphView.html",
    "views/knowledgeLibrary.css",
    "views/knowledgeLibrary.html",
    "views/macrosLibrary.css",
    "views/macrosLibrary.html",
    "views/options.css",
    "views/options.html",
    "views/pageKnowledge.css",
    "views/pageKnowledge.html",
    "views/pageMacros.css",
    "views/pageMacros.html",
    "views/pdfView.css",
    "views/pdfView.html",
];

function copyLibraryAssets(outDir) {
    mkdirSync(`${outDir}/views`, { recursive: true });
    for (const asset of libraryAssets) {
        copyFileSync(`${srcDir}/${asset}`, `${outDir}/${asset}`);
    }
}

function copyCommonStaticAssets(outDir) {
    // Copy library assets
    copyLibraryAssets(outDir);

    // Copy other common assets
    mkdirSync(`${outDir}/offscreen`, { recursive: true });
    copyFileSync(
        `${srcDir}/offscreen/offscreen.html`,
        `${outDir}/offscreen/offscreen.html`,
    );

    mkdirSync(`${outDir}/sites`, { recursive: true });
    copyFileSync(
        `${srcDir}/sites/paleobiodbSchema.mts`,
        `${outDir}/sites/paleobiodbSchema.mts`,
    );

    cpSync(`${srcDir}/images`, `${outDir}/images`, { recursive: true });

    // Copy vendor assets
    for (const [src, destRel] of vendorAssets) {
        const dest = resolve(outDir, destRel);
        mkdirSync(dirname(dest), { recursive: true });
        copyFileSync(resolve(__dirname, "../", src), dest);
    }
}

if (verbose)
    console.log(
        chalk.blueBright(
            `\n🔨 Building in ${buildMode.toUpperCase()} mode...\n`,
        ),
    );

//
// ------------------------
// 🔹 Browser Extension
// ------------------------
//
if (verbose) console.log(chalk.cyan("🚀 Building Browser extension..."));

// Service worker (ESM)
await build(
    createBuildOptions(chromeOutDir, {
        emptyOutDir: !isDev,
        rollupOptions: {
            input: { serviceWorker: resolve(srcDir, "serviceWorker/index.ts") },
            output: {
                format: "es",
                entryFileNames: "serviceWorker.js",
            },
        },
    }),
);
if (verbose) console.log(chalk.green("✅ Chrome service worker built"));

// Content scripts (IIFE)
for (const [name, relPath] of Object.entries(sharedScripts)) {
    const input = resolve(srcDir, relPath);
    if (verbose) console.log(chalk.yellow(`➡️  Chrome content: ${name}`));
    await build(
        createBuildOptions(chromeOutDir, {
            rollupOptions: {
                input,
                output: {
                    format: "iife",
                    entryFileNames: `${name}.js`,
                    inlineDynamicImports: true,
                },
            },
        }),
    );
    if (verbose) console.log(chalk.green(`✅ Chrome ${name}.js built`));
}

// Static file copy
if (verbose) console.log(chalk.cyan("\n📁 Copying Chrome static files..."));
copyFileSync(`${srcDir}/manifest.json`, `${chromeOutDir}/manifest.json`);
copyCommonStaticAssets(chromeOutDir);
if (verbose) console.log(chalk.green("✅ Chrome static assets copied"));

//
// ------------------------
// 🟣 Electron Extension
// ------------------------
//
if (verbose) console.log(chalk.cyan("\n🚀 Building Electron extension..."));

for (const [name, relPath] of Object.entries(sharedScripts)) {
    const input = resolve(srcDir, relPath);
    if (verbose) console.log(chalk.yellow(`➡️  Electron shared: ${name}`));
    await build(
        createBuildOptions(electronOutDir, {
            rollupOptions: {
                input,
                output: {
                    format: "iife",
                    entryFileNames: `${name}.js`,
                    inlineDynamicImports: true,
                },
            },
        }),
    );
    if (verbose) console.log(chalk.green(`✅ Electron ${name}.js built`));
}

for (const [name, relPath] of Object.entries(electronOnlyScripts)) {
    const input = resolve(__dirname, relPath);
    if (verbose) console.log(chalk.yellow(`➡️  Electron only: ${name}`));
    await build(
        createBuildOptions(electronOutDir, {
            rollupOptions: {
                input,
                output: {
                    format: "iife",
                    entryFileNames: `${name}.js`,
                    inlineDynamicImports: true,
                },
            },
        }),
    );
    if (verbose) console.log(chalk.green(`✅ Electron ${name}.js built`));
}

// Copy electron manifest and assets
if (verbose) console.log(chalk.cyan("\n📁 Copying Electron static files..."));
copyFileSync(
    `${electronSrcDir}/manifest.json`,
    `${electronOutDir}/manifest.json`,
);
copyCommonStaticAssets(electronOutDir);
if (verbose) console.log(chalk.green("✅ Electron static assets copied\n"));

// Update build hash to mark successful completion
// updateBuildHash(true); // true = actually built something

if (verbose)
    console.log(
        chalk.bold.green(
            `\n🎉 Extension build complete [${buildMode.toUpperCase()} mode]`,
        ),
    );
