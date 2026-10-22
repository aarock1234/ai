// Pi extension of the Go Pi harness adapter. Pi loads it with --extension
// inside the sandbox; it is the in-sandbox half of the adapter:
//
//   - It replaces Pi's built-in file and shell tools with the adapter's
//     declarations (the upstream @ai-sdk/harness-pi schemas), delegating to
//     Pi's native implementations, so that built-in tools are filtered,
//     approved and confined to the configured paths.
//   - It registers the application tools of each turn. Their calls are
//     answered by the host through the RPC extension UI protocol: the tool
//     asks an input dialog whose response carries the tool result.
//   - It asks the host for built-in tool approvals the same way.
//   - It applies each turn's instructions to the system prompt.
//   - It registers the provider configurations and MCP servers the host
//     resolved.
//
// The host writes the configuration file named by AI_SDK_HARNESS_PI_CONFIG
// before Pi starts, and the turn file it references before every turn.

import { readFileSync, realpathSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  createBashToolDefinition,
  createEditToolDefinition,
  createFindToolDefinition,
  createGrepToolDefinition,
  createLsToolDefinition,
  createReadToolDefinition,
  createWriteToolDefinition,
  DEFAULT_MAX_BYTES,
  formatSize,
  truncateHead,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const DIALOG_TITLE = "ai-sdk-harness-pi";
const SYNC_COMMAND = "ai-sdk-harness-sync";
const FILE_CHANGE_STATUS = "ai-sdk-harness-pi:file-change";
const STARTUP_ERROR_COMMAND = "ai-sdk-harness-startup-error";
const HOST_TOOL_CONTINUATION = "Call the tool again with narrower parameters to inspect the omitted output.";

const config = JSON.parse(readFileSync(process.env.AI_SDK_HARNESS_PI_CONFIG, "utf8"));

export default function harnessExtension(pi) {
  const startupErrors = [];
  const paths = createPathPolicy(config);
  const registeredHostTools = new Set();

  for (const provider of config.providers ?? []) {
    try {
      pi.registerProvider(provider.name, provider.config);
    } catch (error) {
      startupErrors.push(`Pi provider ${JSON.stringify(provider.name)}: ${errorText(error)}`);
    }
  }

  for (const server of config.mcpServers ?? []) {
    try {
      pi.registerMcpServer(server.name, server.config);
    } catch (error) {
      startupErrors.push(`Pi MCP server ${JSON.stringify(server.name)}: ${errorText(error)}`);
    }
  }

  for (const nativeName of config.builtinTools ?? []) {
    pi.registerTool(createBuiltinTool({ nativeName, paths }));
  }

  applyHostTools({
    pi,
    tools: readTurn().tools ?? [],
    registered: registeredHostTools,
  });

  // The host lists the commands when Pi starts and fails the session start
  // when this command reports registration errors.
  if (startupErrors.length > 0) {
    pi.registerCommand(STARTUP_ERROR_COMMAND, {
      description: startupErrors.join("\n"),
      handler: async () => {},
    });
  }

  pi.on("before_agent_start", (event) => {
    event.systemPromptOptions.appendSystemPrompt = readTurn().instructions ?? "";
  });

  pi.registerCommand(SYNC_COMMAND, {
    description: "Apply the tools of the next harness turn.",
    handler: async (args, ctx) => {
      if (args.trim() === "reload") {
        await ctx.reload();

        return;
      }

      applyHostTools({
        pi,
        tools: readTurn().tools ?? [],
        registered: registeredHostTools,
      });
    },
  });
}

function readTurn() {
  try {
    return JSON.parse(readFileSync(config.turnFile, "utf8"));
  } catch {
    return {};
  }
}

// applyHostTools registers the turn's application tools and withdraws the
// tools of earlier turns that the turn no longer offers.
function applyHostTools({ pi, tools, registered }) {
  const names = new Set(tools.map((tool) => tool.name));

  for (const name of registered) {
    if (!names.has(name)) {
      pi.registerTool(createWithdrawnTool(name));
      registered.delete(name);
    }
  }

  for (const tool of tools) {
    pi.registerTool(createHostTool(tool));
    registered.add(tool.name);
  }
}

function createHostTool(spec) {
  const schema = spec.inputSchema ?? {
    type: "object",
    properties: {},
    additionalProperties: true,
  };

  return {
    name: spec.name,
    label: spec.name,
    description: spec.description || `User-registered tool ${spec.name}`,
    parameters: Type.Unsafe(schema),
    async execute(toolCallId, _params, signal, _onUpdate, ctx) {
      const response = await askHost({
        ctx,
        signal,
        request: {
          kind: "tool-result",
          toolCallId,
          toolName: spec.name,
        },
      });

      if (response === undefined) {
        throw new Error("Pi tool call ended before the host delivered its result.");
      }

      return toPiHostToolResult(response);
    },
  };
}

function createWithdrawnTool(name) {
  return {
    name,
    label: name,
    description: `Withdrawn tool ${name}`,
    exposure: "hidden",
    parameters: Type.Object({}),
    async execute() {
      throw new Error(`Tool ${name} is not available in this turn.`);
    },
  };
}

// toPiHostToolResult converts the host's result submission. Results built
// from a model-facing tool output keep their text and images; raw outputs
// arrive serialized as one text part. An erroring model-facing output fails
// the call, as upstream.
function toPiHostToolResult(submission) {
  if (!submission.fromModelOutput) {
    const text = submission.content.map((part) => part.text ?? "").join("");

    return textResult(truncateHeadWithNotice(text, HOST_TOOL_CONTINUATION));
  }

  const content = truncateModelOutput(submission.content, HOST_TOOL_CONTINUATION);

  if (submission.isError) {
    const message = content
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join("\n");

    throw new Error(message);
  }

  return {
    content,
    details: undefined,
  };
}

// askHost sends a request to the host through an input dialog and decodes
// the JSON response. It returns undefined when the dialog was cancelled or
// the call was aborted.
async function askHost({ ctx, signal, request }) {
  const placeholder = JSON.stringify(request);
  const value = await ctx.ui.input(DIALOG_TITLE, placeholder, {
    signal,
  });

  if (value === undefined) {
    return undefined;
  }

  return JSON.parse(value);
}

const BUILTIN_DEFINITIONS = {
  read: {
    kind: "readonly",
    description: (cwd) => createReadToolDefinition(cwd).description,
    parameters: Type.Object({
      file_path: Type.String(),
      offset: Type.Optional(
        Type.Integer({
          minimum: 1,
          description: "Line number to start reading from (1-indexed).",
        }),
      ),
      limit: Type.Optional(
        Type.Integer({
          minimum: 1,
          description: "Maximum number of lines to read.",
        }),
      ),
    }),
    native: createReadToolDefinition,
    access: "read",
    pathOf: (params) => params.file_path,
    toNative: (params, filePath) => ({
      path: filePath,
      offset: params.offset,
      limit: params.limit,
    }),
  },
  write: {
    kind: "edit",
    description: () => "Write content to a file.",
    parameters: Type.Object({
      file_path: Type.String(),
      content: Type.String(),
    }),
    native: createWriteToolDefinition,
    access: "write",
    fileChange: true,
    pathOf: (params) => params.file_path,
    toNative: (params, filePath) => ({
      path: filePath,
      content: params.content,
    }),
  },
  edit: {
    kind: "edit",
    description: () => "Edit a file by exact-string replacement.",
    parameters: Type.Object({
      file_path: Type.String(),
      old_string: Type.String(),
      new_string: Type.String(),
    }),
    native: createEditToolDefinition,
    access: "write",
    fileChange: true,
    pathOf: (params) => params.file_path,
    toNative: (params, filePath) => ({
      path: filePath,
      edits: [
        {
          oldText: params.old_string,
          newText: params.new_string,
        },
      ],
    }),
  },
  bash: {
    kind: "bash",
    description: () => "Execute a shell command.",
    parameters: Type.Object({
      command: Type.String(),
      timeout: Type.Optional(
        Type.Number({
          description: "Timeout in seconds.",
        }),
      ),
    }),
    native: createBashToolDefinition,
    access: "none",
    toNative: (params) => ({
      command: params.command,
      timeout: params.timeout,
    }),
  },
  grep: {
    kind: "readonly",
    description: () => "Search file contents with regex.",
    parameters: Type.Object({
      pattern: Type.String(),
      path: Type.Optional(Type.String()),
      glob: Type.Optional(Type.String()),
      ignoreCase: Type.Optional(Type.Boolean()),
      literal: Type.Optional(Type.Boolean()),
      context: Type.Optional(Type.Number()),
      limit: Type.Optional(Type.Number()),
    }),
    native: createGrepToolDefinition,
    access: "read",
    pathOf: (params) => params.path ?? ".",
    toNative: (params, searchPath) => ({
      ...params,
      path: searchPath,
    }),
  },
  find: {
    kind: "readonly",
    description: () => "Find files matching a glob pattern.",
    parameters: Type.Object({
      pattern: Type.String(),
      path: Type.Optional(Type.String()),
      limit: Type.Optional(Type.Number()),
    }),
    native: createFindToolDefinition,
    access: "read",
    pathOf: (params) => params.path ?? ".",
    toNative: (params, searchPath) => ({
      ...params,
      path: searchPath,
    }),
  },
  ls: {
    kind: "readonly",
    description: () => "List directory entries.",
    parameters: Type.Object({
      path: Type.Optional(Type.String()),
      limit: Type.Optional(Type.Number()),
    }),
    native: createLsToolDefinition,
    access: "read",
    pathOf: (params) => params.path ?? ".",
    toNative: (params, listPath) => ({
      ...params,
      path: listPath,
    }),
  },
};

// createBuiltinTool declares one built-in tool with the adapter's schema and
// runs it through Pi's native implementation after the approval and path
// checks.
function createBuiltinTool({ nativeName, paths }) {
  const definition = BUILTIN_DEFINITIONS[nativeName];
  if (definition === undefined) {
    throw new Error(`Unknown Pi built-in tool ${nativeName}`);
  }

  const native = definition.native(config.cwd);

  return {
    name: nativeName,
    label: nativeName,
    description: definition.description(config.cwd),
    parameters: definition.parameters,
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const denial = await requestApproval({
        ctx,
        signal,
        toolCallId,
        nativeName,
        kind: definition.kind,
      });
      if (denial !== undefined) {
        return denial;
      }

      const target = definition.pathOf?.(params);
      const resolvedPath = target === undefined ? undefined : paths.check(target, definition.access);
      const existedBefore = definition.fileChange ? paths.exists(resolvedPath) : false;

      const result = await native.execute(
        toolCallId,
        definition.toNative(params, resolvedPath),
        signal,
        onUpdate,
        ctx,
      );

      if (definition.fileChange && paths.isWorkspacePath(resolvedPath)) {
        reportFileChange({
          ctx,
          event: nativeName === "write" && !existedBefore ? "create" : "modify",
          path: paths.relativeToWorkspace(resolvedPath),
        });
      }

      return result;
    },
  };
}

// requestApproval asks the host to approve a built-in tool call when the
// permission mode requires it. It returns the denial result, or undefined
// when the call may run.
async function requestApproval({ ctx, signal, toolCallId, nativeName, kind }) {
  if (!requiresApproval(kind)) {
    return undefined;
  }

  const decision = await askHost({
    ctx,
    signal,
    request: {
      kind: "tool-approval",
      toolCallId,
      toolName: nativeName,
    },
  });

  if (decision?.approved === true) {
    return undefined;
  }

  return textResult(
    JSON.stringify({
      type: "execution-denied",
      reason: decision?.reason,
    }),
  );
}

function requiresApproval(kind) {
  switch (config.permissionMode) {
    case "allow-edits":
      return kind === "bash";
    case "allow-reads":
      return kind === "edit" || kind === "bash";
    default:
      return false;
  }
}

function reportFileChange({ ctx, event, path: filePath }) {
  ctx.ui.setStatus(
    FILE_CHANGE_STATUS,
    JSON.stringify({
      event,
      path: filePath,
    }),
  );
}

// createPathPolicy confines the native file tools to the workspace, the
// harness skills directory (read-only) and the configured roots. Denied
// roots take precedence, including through symbolic links: paths are
// compared both as written and after resolving the links of their longest
// existing prefix. The bash tool is not restricted.
function createPathPolicy({ cwd, skillsDir, pathPolicy }) {
  const policy = pathPolicy ?? {};
  const workspace = canonicalize(path.resolve(cwd));
  const readableRoots = [skillsDir, ...(policy.readableRoots ?? [])].filter(Boolean).map(normalizeRoot);
  const writableRoots = (policy.writableRoots ?? []).map(normalizeRoot);
  const deniedRoots = (policy.deniedRoots ?? []).map((root) => path.posix.normalize(root));
  const canonicalDeniedRoots = [...new Set([...deniedRoots, ...deniedRoots.map(canonicalize)])];

  function resolveInput(inputPath) {
    const expanded = inputPath === "~" || inputPath.startsWith("~/") ? path.join(os.homedir(), inputPath.slice(1)) : inputPath;

    return path.resolve(cwd, expanded);
  }

  function check(inputPath, access) {
    const resolved = resolveInput(inputPath);
    if (access === "none") {
      return resolved;
    }

    const canonical = canonicalize(resolved);

    if (canonicalDeniedRoots.some((root) => isInside(root, resolved) || isInside(root, canonical))) {
      throw new Error(`Pi path is inside a denied root: ${inputPath}`);
    }

    const allowedRoots = access === "write" ? [workspace, ...writableRoots] : [workspace, ...readableRoots, ...writableRoots];

    if (!allowedRoots.some((root) => isInside(root, canonical))) {
      const scope = access === "write" ? "the workspace" : "the readable roots";
      throw new Error(`Pi path escapes ${scope}: ${inputPath}`);
    }

    return resolved;
  }

  return {
    check,
    exists(filePath) {
      try {
        realpathSync.native(filePath);

        return true;
      } catch {
        return false;
      }
    },
    isWorkspacePath(filePath) {
      return filePath !== undefined && isInside(workspace, canonicalize(filePath));
    },
    relativeToWorkspace(filePath) {
      return path.relative(workspace, canonicalize(filePath)) || ".";
    },
  };
}

function normalizeRoot(root) {
  if (!path.posix.isAbsolute(root)) {
    throw new Error(`Pi path policy root must be an absolute sandbox path: ${root}`);
  }

  return canonicalize(path.posix.normalize(root));
}

// canonicalize resolves the symbolic links of the longest existing prefix of
// a path.
function canonicalize(inputPath) {
  try {
    return realpathSync.native(inputPath);
  } catch {
    const parent = path.dirname(inputPath);
    if (parent === inputPath) {
      return inputPath;
    }

    return path.join(canonicalize(parent), path.basename(inputPath));
  }
}

function isInside(parent, candidate) {
  const relative = path.relative(parent, candidate);

  return relative === "" || (relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
}

function textResult(text) {
  return {
    content: [
      {
        type: "text",
        text,
      },
    ],
    details: null,
  };
}

function errorText(error) {
  return error instanceof Error ? error.message : String(error);
}

// The truncation helpers port upstream pi-tool-result.ts.

function joinContentAndNotice(content, notice) {
  return content.length > 0 ? `${content}\n\n[${notice}]` : `[${notice}]`;
}

function truncateUtf8Start(text, maxBytes) {
  const buffer = Buffer.from(text, "utf8");
  if (buffer.length <= maxBytes) {
    return text;
  }

  let end = maxBytes;
  while (end > 0 && (buffer[end] & 0xc0) === 0x80) {
    end--;
  }

  return buffer.subarray(0, end).toString("utf8");
}

function headTruncation(text, continuation) {
  const truncation = truncateHead(text);
  if (!truncation.truncated) {
    return {
      retainedText: text,
    };
  }

  if (truncation.firstLineExceedsLimit) {
    const content = truncateUtf8Start(text, DEFAULT_MAX_BYTES);

    return {
      retainedText: content,
      notice: `Output truncated: showing the first ${formatSize(Buffer.byteLength(content, "utf8"))} of a ${formatSize(truncation.totalBytes)} line. ${continuation}`,
    };
  }

  const limit = truncation.truncatedBy === "bytes" ? ` (${formatSize(truncation.maxBytes)} limit)` : "";

  return {
    retainedText: truncation.content,
    notice: `Output truncated: showing the first ${truncation.outputLines} of ${truncation.totalLines} lines${limit}. ${continuation}`,
  };
}

function truncateHeadWithNotice(text, continuation) {
  const { retainedText, notice } = headTruncation(text, continuation);

  return notice === undefined ? retainedText : joinContentAndNotice(retainedText, notice);
}

function truncateModelOutput(content, continuation) {
  const text = content
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n");
  const { retainedText, notice } = headTruncation(text, continuation);
  if (notice === undefined) {
    return content;
  }

  const truncated = [];
  let offset = 0;
  let lastTextPart;

  for (const part of content) {
    if (part.type === "image") {
      truncated.push(part);

      continue;
    }

    const prefix = retainedText.slice(offset, offset + part.text.length);
    offset += part.text.length + 1;

    if (prefix.length > 0) {
      lastTextPart = {
        ...part,
        text: prefix,
      };
      truncated.push(lastTextPart);
    }
  }

  if (lastTextPart !== undefined) {
    lastTextPart.text = joinContentAndNotice(lastTextPart.text, notice);
  } else {
    truncated.splice(
      content.findIndex((part) => part.type === "text"),
      0,
      {
        type: "text",
        text: joinContentAndNotice("", notice),
      },
    );
  }

  return truncated;
}
