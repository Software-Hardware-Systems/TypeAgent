// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import { CommandHandlerContext } from "../context/commandHandlerContext.js";

import {
    CommandDescriptor,
    FlagDefinitions,
    ParameterDefinitions,
    ParsedCommandParams,
    CompletionGroup,
} from "@typeagent/agent-sdk";
import {
    getFlagMultiple,
    getFlagType,
    resolveFlag,
} from "@typeagent/agent-sdk/helpers/command";
import { parseParams } from "./parameters.js";
import {
    getDefaultSubCommandDescriptor,
    normalizeCommand,
    resolveCommand,
    ResolveCommandResult,
} from "./command.js";

import registerDebug from "debug";
const debug = registerDebug("typeagent:command:completion");
const debugError = registerDebug("typeagent:command:completion:error");

export type CommandCompletionResult = {
    startIndex: number; // the index for the input where completion starts
    completions: CompletionGroup[]; // All the partial completions available after partial (and space if true)
    space: boolean; // require space before the completion   (e.g. false if we are trying to complete a command)
};

// Return the index of the last incomplete term for completion.
// if the last term is the '@' command itself, return the index right after the '@'.
// Input with trailing space doesn't have incomplete term, so return -1.
function getCompletionStartIndex(input: string) {
    const commandPrefix = input.match(/^\s*@/);
    if (commandPrefix !== null) {
        // Input is a command
        const command = input.substring(commandPrefix.length);
        if (!/\s/.test(command)) {
            // No space on command yet just return right after the '@' as the start of the last term.
            return commandPrefix.length;
        }
    }

    const suffix = input.match(/\s\S+$/);
    return suffix !== null ? input.length - suffix[0].length : -1;
}

// Return the full flag name if we are waiting a flag value.  Add boolean values for completions and return undefined if the flag is boolean.
function getPendingFlag(
    params: ParsedCommandParams<ParameterDefinitions>,
    flags: FlagDefinitions | undefined,
    completions: CompletionGroup[],
) {
    if (params.tokens.length === 0 || flags === undefined) {
        return undefined;
    }
    const lastToken = params.tokens[params.tokens.length - 1];
    const resolvedFlag = resolveFlag(flags, lastToken);
    if (resolvedFlag === undefined) {
        return undefined;
    }
    const type = getFlagType(resolvedFlag[1]);
    if (type === "boolean") {
        completions.push({
            name: `--${resolvedFlag[0]}`,
            completions: ["true", "false"],
        });
        return undefined; // doesn't require a value.
    }
    if (type === "json") {
        return lastToken;
    }

    return `--${resolvedFlag[0]}`; // use the full flag name in case it was a short flag
}

// True if surrounded by quotes at both ends (matching single or double quotes).
// False if only start with a quote.
// Undefined if no starting quote.

function isFullyQuoted(value: string) {
    const len = value.length;
    if (len === 0) {
        return undefined;
    }
    const firstChar = value[0];
    if (firstChar !== "'" && firstChar !== '"') {
        return undefined;
    }

    return (
        len > 1 &&
        value[len - 1] === firstChar &&
        !(len > 2 && value[len - 2] === "\\")
    );
}

function collectFlags(
    agentCommandCompletions: string[],
    flags: FlagDefinitions,
    parsedFlags: any,
) {
    const flagCompletions: string[] = [];
    for (const [key, value] of Object.entries(flags)) {
        const multiple = getFlagMultiple(value);
        if (!multiple) {
            if (getFlagType(value) === "json") {
                // JSON property flags
                agentCommandCompletions.push(`--${key}.`);
            }
            if (parsedFlags?.[key] !== undefined) {
                // filter out non-multiple flags that is already set.
                continue;
            }
        }
        flagCompletions.push(`--${key}`);
        if (value.char !== undefined) {
            flagCompletions.push(`-${value.char}`);
        }
    }

    return flagCompletions;
}

async function getCommandParameterCompletion(
    descriptor: CommandDescriptor,
    context: CommandHandlerContext,
    result: ResolveCommandResult,
): Promise<CompletionGroup[] | undefined> {
    const completions: CompletionGroup[] = [];
    if (typeof descriptor.parameters !== "object") {
        // No more completion, return undefined;
        return undefined;
    }
    const flags = descriptor.parameters.flags;
    const params = parseParams(result.suffix, descriptor.parameters, true);
    const pendingFlag = getPendingFlag(params, flags, completions);
    const agentCommandCompletions: string[] = [];
    if (pendingFlag === undefined) {
        // TODO: auto inject boolean value for boolean args.
        agentCommandCompletions.push(...params.nextArgs);
        if (flags !== undefined) {
            const flagCompletions = collectFlags(
                agentCommandCompletions,
                flags,
                params.flags,
            );
            if (flagCompletions.length > 0) {
                completions.push({
                    name: "Command Flags",
                    completions: flagCompletions,
                });
            }
        }
    } else {
        // get the potential values for the pending flag
        agentCommandCompletions.push(pendingFlag);
    }

    const agent = context.agents.getAppAgent(result.actualAppAgentName);
    if (agent.getCommandCompletion) {
        const { tokens, lastCompletableParam, lastParamImplicitQuotes } =
            params;

        if (lastCompletableParam !== undefined && tokens.length > 0) {
            const valueToken = tokens[tokens.length - 1];
            const quoted = isFullyQuoted(valueToken);
            if (
                quoted === false ||
                (quoted === undefined && lastParamImplicitQuotes)
            ) {
                agentCommandCompletions.push(lastCompletableParam);
            }
        }
        if (agentCommandCompletions.length > 0) {
            const sessionContext = context.agents.getSessionContext(
                result.actualAppAgentName,
            );
            completions.push(
                ...(await agent.getCommandCompletion(
                    result.commands,
                    params,
                    agentCommandCompletions,
                    sessionContext,
                )),
            );
        }
    }
    return completions;
}

export async function getCommandCompletion(
    input: string,
    context: CommandHandlerContext,
): Promise<CommandCompletionResult | undefined> {
    try {
        debug(`Command completion start: '${input}'`);
        const completionStartIndex = getCompletionStartIndex(input);
        const commandPrefix =
            completionStartIndex !== -1
                ? input.substring(0, completionStartIndex)
                : input;

        // Trim spaces and remove leading '@'
        const partialCommand = normalizeCommand(commandPrefix, context);

        debug(`Command completion resolve command: '${partialCommand}'`);
        const result = await resolveCommand(partialCommand, context);

        const table = result.table;
        if (table === undefined) {
            // Unknown app agent, or appAgent doesn't support commands
            // Return undefined to indicate no more completions for this prefix.
            return undefined;
        }

        // Collect completions
        const completions: CompletionGroup[] = [];
        if (commandPrefix.trim() === "") {
            completions.push({
                name: "Command Prefixes",
                completions: ["@"],
            });
        }

        const descriptor = result.descriptor;
        if (descriptor !== undefined) {
            if (
                result.suffix.length === 0 &&
                table !== undefined &&
                getDefaultSubCommandDescriptor(table) === result.descriptor
            ) {
                // Match the default sub command.  Includes additional subcommand names
                completions.push({
                    name: "Subcommands",
                    completions: Object.keys(table.commands),
                });
            }
            const parameterCompletions = await getCommandParameterCompletion(
                descriptor,
                context,
                result,
            );
            if (parameterCompletions === undefined) {
                if (completions.length === 0) {
                    // No more completion, return undefined;
                    return undefined;
                }
            } else {
                completions.push(...parameterCompletions);
            }
        } else {
            if (result.suffix.length !== 0) {
                // Unknown command
                // Return undefined to indicate no more completions for this prefix.
                return undefined;
            }
            completions.push({
                name: "Subcommands",
                completions: Object.keys(table.commands),
            });
            if (
                result.parsedAppAgentName === undefined &&
                result.commands.length === 0
            ) {
                // Include the agent names
                completions.push({
                    name: "Agent Names",
                    completions: context.agents
                        .getAppAgentNames()
                        .filter((name) =>
                            context.agents.isCommandEnabled(name),
                        ),
                });
            }
        }

        const space =
            completionStartIndex > 0 && input[completionStartIndex - 1] !== "@";
        const completionResult = {
            startIndex: completionStartIndex,
            completions,
            space,
        };

        debug(`Command completion result:`, completionResult);
        return completionResult;
    } catch (e: any) {
        debugError(`Command completion error: ${e}\n${e.stack}`);
        return undefined;
    }
}
