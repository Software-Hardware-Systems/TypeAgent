// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import { getFileName, readAllText } from "typeagent";
import {
    DocMemory,
    DocMemorySettings,
    DocPart,
    DocPartMeta,
} from "./docMemory.js";
import { splitLargeTextIntoChunks } from "knowledge-processor";
import * as tp from "textpro";
import { parseVttTranscript } from "./transcript.js";
import { filePathToUrlString } from "memory-storage";
import path from "path";
import { getHtml } from "aiclient";
import { Result, success } from "typechat";

/**
 * Import a text document as DocMemory
 * You must call buildIndex before you can query the memory
 *
 * Uses file extensions to determine how to import.
 *  default: treat as text
 *  .html => parse html
 *  .vtt => parse vtt transcript
 * @param docFilePath
 * @param maxCharsPerChunk
 * @param docName
 * @param settings
 * @returns
 */
export async function importTextFile(
    docFilePath: string,
    maxCharsPerChunk: number,
    docName?: string,
    settings?: DocMemorySettings,
): Promise<DocMemory> {
    const docText = await readAllText(docFilePath);
    docName ??= getFileName(docFilePath);
    const ext = path.extname(docFilePath);

    const sourceUrl = filePathToUrlString(docFilePath);
    let parts: DocPart[];
    switch (ext) {
        default:
            parts = docPartsFromText(docText, maxCharsPerChunk, sourceUrl);
            break;
        case ".html":
        case ".htm":
            parts = docPartsFromHtml(
                docText,
                true,
                maxCharsPerChunk,
                sourceUrl,
            );
            break;
        case ".vtt":
            parts = docPartsFromVtt(docText, sourceUrl);
            if (parts.length > 0) {
                parts = mergeDocParts(
                    parts,
                    parts[0].metadata,
                    maxCharsPerChunk,
                );
            }
            break;
        case ".md":
            parts = docPartsFromMarkdown(docText, maxCharsPerChunk, sourceUrl);
            break;
    }
    return new DocMemory(docName, parts, settings);
}

/**
 * Import a web page as DocMemory
 * You must call buildIndex before you can query the memory
 * @param url
 * @param maxCharsPerChunk
 * @param settings
 * @returns
 */
export async function importWebPage(
    url: string,
    maxCharsPerChunk: number,
    settings?: DocMemorySettings,
): Promise<Result<DocMemory>> {
    const htmlResult = await getHtml(url);
    if (!htmlResult.success) {
        return htmlResult;
    }
    const parts = docPartsFromHtml(
        htmlResult.data,
        false,
        maxCharsPerChunk,
        url,
    );
    const docMemory = new DocMemory(url, parts, settings);
    return success(docMemory);
}
/**
 * Import the given text as separate blocks
 * @param documentText
 * @param maxCharsPerChunk
 * @param sourceUrl
 * @returns
 */
export function docPartsFromText(
    documentText: string,
    maxCharsPerChunk: number,
    sourceUrl?: string,
): DocPart[] {
    const blocks: DocPart[] = [];
    for (const chunk of splitLargeTextIntoChunks(
        documentText,
        maxCharsPerChunk,
        false,
    )) {
        const block = new DocPart(chunk, new DocPartMeta(sourceUrl));
        blocks.push(block);
    }
    return blocks;
}

/**
 * Import the text as a single DocBlock with multiple chunks
 * @param documentText
 * @param maxCharsPerChunk
 * @param sourceUrl
 * @returns
 */
export function docPartFromText(
    documentText: string,
    maxCharsPerChunk: number,
    sourceUrl?: string,
): DocPart {
    const textChunks = [
        ...splitLargeTextIntoChunks(documentText, maxCharsPerChunk, false),
    ];
    return new DocPart(textChunks, new DocPartMeta(sourceUrl));
}

/**
 * Break the given html into DocParts
 * @param html html text
 * @param textOnly if true, use only text, removing all formatting etc.
 * @param maxCharsPerChunk
 * @param sourceUrl
 * @param rootTag
 * @returns
 */
export function docPartsFromHtml(
    html: string,
    textOnly: boolean,
    maxCharsPerChunk: number,
    sourceUrl?: string,
    rootTag?: string,
): DocPart[] {
    if (textOnly) {
        const htmlText = tp.htmlToText(html);
        return docPartsFromText(htmlText, maxCharsPerChunk, sourceUrl);
    } else {
        const markdown = tp.htmlToMarkdown(html, rootTag);
        return docPartsFromMarkdown(markdown, maxCharsPerChunk, sourceUrl);
    }
}

/**
 * Convert markdown text into DocParts. These can be added to DocMemory
 * @param markdown
 * @param maxCharsPerChunk
 * @param sourceUrl
 * @returns
 */
export function docPartsFromMarkdown(
    markdown: string,
    maxCharsPerChunk?: number,
    sourceUrl?: string,
): DocPart[] {
    const [textBlocks, knowledgeBlocks] = tp.textAndKnowledgeBlocksFromMarkdown(
        markdown,
        maxCharsPerChunk,
    );
    if (textBlocks.length !== knowledgeBlocks.length) {
        throw new Error(
            `textBlocks.length ${textBlocks.length} !== knowledgeBlocks.length ${knowledgeBlocks.length}`,
        );
    }
    const parts: DocPart[] = [];
    for (let i = 0; i < textBlocks.length; ++i) {
        const kBlock = knowledgeBlocks[i];
        const part = new DocPart(
            textBlocks[i],
            new DocPartMeta(sourceUrl),
            kBlock.tags.size > 0 ? [...kBlock.tags.values()] : undefined,
            undefined,
            kBlock.knowledge,
        );
        if (kBlock.sTags && kBlock.sTags.length > 0) {
            part.sTags = kBlock.sTags;
        }
        parts.push(part);
    }
    return parts;
}

/**
 * Parse a VTT document as a set of document parts
 * @param transcriptText
 * @returns
 */
export function docPartsFromVtt(
    transcriptText: string,
    sourceUrl?: string,
): DocPart[] {
    const [parts, _] = parseVttTranscript<DocPart>(
        transcriptText,
        new Date(),
        (speaker: string) => new DocPart([], new DocPartMeta(sourceUrl)),
    );
    return parts;
}

/**
 * Combine small DocParts into larger ones
 * @param parts
 * @param metadata
 * @param maxCharsPerChunk
 * @returns
 */
export function mergeDocParts(
    parts: DocPart[],
    metadata: DocPartMeta,
    maxCharsPerChunk: number,
): DocPart[] {
    const allChunks = parts.flatMap((p) => p.textChunks);
    const mergedChunks: DocPart[] = [];
    // This will merge all small chunks into larger chunks as needed.. but not exceed
    // maxCharsPerChunk
    for (const chunk of splitLargeTextIntoChunks(
        allChunks,
        maxCharsPerChunk,
        true,
    )) {
        mergedChunks.push(new DocPart(chunk, metadata));
    }
    return mergedChunks;
}
