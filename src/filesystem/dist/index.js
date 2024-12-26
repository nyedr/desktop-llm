/**
 * MCP Filesystem Server
 * This server implements the Model Context Protocol for filesystem operations.
 * It provides secure access to the filesystem with directory restrictions and validation.
 *
 * Implementation Notes:
 * 1. Server Initialization Flow:
 *    - Client sends 'initialize' request with capabilities
 *    - Server responds with supported capabilities
 *    - Client sends 'initialized' notification
 *    - Server starts accepting other requests
 *
 * 2. Capability Handling:
 *    - Server advertises filesystem tools
 *    - Each tool requires specific capabilities
 *    - Server validates client capabilities before operations
 *
 * 3. Protocol Version:
 *    - Server supports multiple protocol versions
 *    - Version negotiation happens during initialization
 *
 * 4. Error Handling:
 *    - All filesystem operations are validated
 *    - Path traversal is prevented
 *    - Access is restricted to allowed directories
 */
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { LATEST_PROTOCOL_VERSION, } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import * as fs from "fs";
import * as path from "path";
const serverState = {
    isShuttingDown: false,
    activeConnections: 0,
    serverStartTime: null,
    initializationTimeout: null,
    allowedDirectories: new Set(),
};
/**
 * Debug logging function that writes to stderr to avoid interfering with stdio transport
 * @param level - Log level (debug, info, warn, error)
 * @param message - Message to log
 * @param data - Optional data to log
 */
function log(level, message, data) {
    const timestamp = new Date().toISOString();
    const uptime = serverState.serverStartTime
        ? `[Uptime: ${Math.floor((Date.now() - serverState.serverStartTime.getTime()) / 1000)}s]`
        : "";
    const prefix = `[Node.js ${process.pid}] [${level.toUpperCase()}] [${timestamp}] ${uptime}`;
    console.error(`${prefix} ${message}`);
    if (data && level !== "debug") {
        console.error(`${prefix} Data:`, typeof data === "string" ? data : JSON.stringify(data, null, 2));
    }
}
// Create transport with debug logging
const transport = new StdioServerTransport();
/**
 * MCP Message Handlers
 * These functions handle the core MCP protocol messages
 */
/**
 * Sends a JSON-RPC result
 * @param id - The request ID
 * @param result - The result to send
 */
async function sendResult(id, result) {
    await transport.send({
        jsonrpc: "2.0",
        id,
        result,
    });
}
/**
 * Sends a JSON-RPC error
 * @param id - The request ID
 * @param code - The error code
 * @param message - The error message
 */
async function sendError(id, code, message) {
    await transport.send({
        jsonrpc: "2.0",
        id,
        error: {
            code,
            message,
        },
    });
}
/**
 * Handles ping requests
 * @param request - The ping request
 */
async function handlePing(request) {
    await sendResult(request.id, {
        status: "ok",
        server_info: {
            pid: process.pid,
            version: process.version,
            uptime: process.uptime(),
            startTime: serverState.serverStartTime?.toISOString(),
            activeConnections: serverState.activeConnections,
        },
    });
}
// Add allowed directories tracking
let allowedDirectories = new Set();
// Add directory validation
async function isPathAllowed(filePath) {
    const absolutePath = path.resolve(filePath);
    return Array.from(serverState.allowedDirectories).some((dir) => absolutePath.startsWith(path.resolve(dir)));
}
/**
 * Validate file path to prevent directory traversal and ensure it exists
 * @param filePath - Path to validate
 * @param shouldExist - Whether the path should exist
 * @returns Normalized absolute path
 */
async function validatePath(filePath, shouldExist = true) {
    try {
        const normalizedPath = path.normalize(filePath);
        const absolutePath = path.resolve(normalizedPath);
        if (shouldExist) {
            const stats = await fs.promises.stat(absolutePath);
            if (!stats) {
                throw new Error(`Path does not exist: ${absolutePath}`);
            }
        }
        return absolutePath;
    }
    catch (error) {
        log("error", `Path validation failed: ${error.message}`, {
            path: filePath,
        });
        throw new Error(`Invalid path: ${error.message}`);
    }
}
// Log startup information
serverState.serverStartTime = new Date();
log("info", "MCP Filesystem Server starting...");
log("info", `Process info - PID: ${process.pid}, Node: ${process.version}, Platform: ${process.platform}`);
log("info", `Working directory: ${process.cwd()}`);
// Add utility function for recursive directory search
async function searchDirectory(dir, pattern, excludePatterns) {
    const results = [];
    const entries = await fs.promises.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        // Check if path should be excluded
        if (excludePatterns?.some((pattern) => fullPath.includes(pattern) || entry.name.includes(pattern))) {
            continue;
        }
        if (entry.isDirectory()) {
            results.push(...(await searchDirectory(fullPath, pattern, excludePatterns)));
        }
        else if (entry.isFile() && entry.name.includes(pattern)) {
            results.push(fullPath);
        }
    }
    return results;
}
// Define request schemas
const ReadFileSchema = z.object({
    method: z.literal("read_file"),
    params: z.object({
        path: z.string(),
    }),
});
const ListDirSchema = z.object({
    method: z.literal("list_dir"),
    params: z.object({
        path: z.string(),
    }),
});
const WriteFileSchema = z.object({
    method: z.literal("write_file"),
    params: z.object({
        path: z.string(),
        content: z.string(),
    }),
});
const MoveFileSchema = z.object({
    method: z.literal("move_file"),
    params: z.object({
        source: z.string(),
        destination: z.string(),
    }),
});
const SearchFilesSchema = z.object({
    method: z.literal("search_files"),
    params: z.object({
        pattern: z.string(),
        excludePatterns: z.array(z.string()).optional(),
    }),
});
const ListAllowedDirectoriesSchema = z.object({
    method: z.literal("list_allowed_directories"),
    params: z.object({}),
});
// Server capabilities
const SERVER_CAPABILITIES = {
    tools: {
        read_file: {
            description: "Read the contents of a file",
            parameters: {
                type: "object",
                properties: {
                    path: { type: "string", description: "Path to the file to read" },
                },
                required: ["path"],
            },
        },
        list_dir: {
            description: "List contents of a directory",
            parameters: {
                type: "object",
                properties: {
                    path: {
                        type: "string",
                        description: "Path to the directory to list",
                    },
                },
                required: ["path"],
            },
        },
        write_file: {
            description: "Write content to a file",
            parameters: {
                type: "object",
                properties: {
                    path: { type: "string", description: "Path to write the file to" },
                    content: { type: "string", description: "Content to write" },
                },
                required: ["path", "content"],
            },
        },
        move_file: {
            description: "Move or rename a file",
            parameters: {
                type: "object",
                properties: {
                    source: { type: "string", description: "Source path" },
                    destination: { type: "string", description: "Destination path" },
                },
                required: ["source", "destination"],
            },
        },
        search_files: {
            description: "Search for files matching a pattern",
            parameters: {
                type: "object",
                properties: {
                    pattern: { type: "string", description: "Search pattern" },
                    excludePatterns: {
                        type: "array",
                        items: { type: "string" },
                        description: "Patterns to exclude",
                    },
                },
                required: ["pattern"],
            },
        },
        list_allowed_directories: {
            description: "List allowed directories",
            parameters: {
                type: "object",
                properties: {},
                required: [],
            },
        },
    },
    version: LATEST_PROTOCOL_VERSION,
    name: "filesystem-server",
    resources: { listChanged: true },
    sampling: { subscribe: true, listChanged: true },
    prompts: { listChanged: true },
    logging: {},
    experimental: {},
};
/**
 * Server implementation details
 * @see Implementation from MCP SDK
 */
const SERVER_IMPLEMENTATION = {
    name: "secure-filesystem-server",
    version: "0.2.0",
};
/**
 * Creates and initializes the MCP server
 * @returns Initialized server instance
 */
async function createServer() {
    const server = new Server(SERVER_IMPLEMENTATION, {
        capabilities: SERVER_CAPABILITIES,
    });
    // Handle initialization completion
    server.oninitialized = () => {
        log("info", "Server initialization complete ✅", {
            clientCapabilities: server.getClientCapabilities(),
            clientVersion: server.getClientVersion(),
        });
    };
    // Set up message handlers for each tool
    server.setRequestHandler(ReadFileSchema, async (request, extra) => {
        try {
            const validatedPath = await validatePath(request.params.path);
            log("debug", `Reading file: ${validatedPath}`);
            const stats = await fs.promises.stat(validatedPath);
            if (!stats.isFile()) {
                throw new Error("Path is not a file");
            }
            const content = await fs.promises.readFile(validatedPath, "utf8");
            log("debug", `Successfully read file: ${validatedPath}`, {
                size: stats.size,
            });
            return { content, size: stats.size };
        }
        catch (error) {
            log("error", `Failed to read file: ${error.message}`, {
                path: request.params.path,
            });
            throw new Error(`Failed to read file: ${error.message}`);
        }
    });
    server.setRequestHandler(ListDirSchema, async (request, extra) => {
        try {
            const validatedPath = await validatePath(request.params.path);
            log("debug", `Listing directory: ${validatedPath}`);
            const stats = await fs.promises.stat(validatedPath);
            if (!stats.isDirectory()) {
                throw new Error("Path is not a directory");
            }
            const files = await fs.promises.readdir(validatedPath, {
                withFileTypes: true,
            });
            const items = files.map((file) => ({
                name: file.name,
                type: file.isDirectory() ? "directory" : "file",
                size: file.isFile()
                    ? fs.statSync(path.join(validatedPath, file.name)).size
                    : null,
                lastModified: fs
                    .statSync(path.join(validatedPath, file.name))
                    .mtime.toISOString(),
            }));
            log("debug", `Successfully listed directory: ${validatedPath}`, {
                itemCount: items.length,
            });
            return { items, totalItems: items.length };
        }
        catch (error) {
            log("error", `Failed to list directory: ${error.message}`, {
                path: request.params.path,
            });
            throw new Error(`Failed to list directory: ${error.message}`);
        }
    });
    server.setRequestHandler(WriteFileSchema, async (request, extra) => {
        try {
            const validatedPath = await validatePath(request.params.path, false);
            if (!(await isPathAllowed(validatedPath))) {
                throw new Error("Path not in allowed directories");
            }
            log("debug", `Writing file: ${validatedPath}`);
            await fs.promises.writeFile(validatedPath, request.params.content, "utf8");
            const stats = await fs.promises.stat(validatedPath);
            return { size: stats.size };
        }
        catch (error) {
            log("error", `Failed to write file: ${error.message}`);
            throw new Error(`Failed to write file: ${error.message}`);
        }
    });
    server.setRequestHandler(MoveFileSchema, async (request, extra) => {
        try {
            const sourcePath = await validatePath(request.params.source);
            const destPath = await validatePath(request.params.destination, false);
            if (!(await isPathAllowed(sourcePath)) ||
                !(await isPathAllowed(destPath))) {
                throw new Error("Path not in allowed directories");
            }
            log("debug", `Moving file: ${sourcePath} to ${destPath}`);
            await fs.promises.rename(sourcePath, destPath);
            return { success: true };
        }
        catch (error) {
            log("error", `Failed to move file: ${error.message}`);
            throw new Error(`Failed to move file: ${error.message}`);
        }
    });
    server.setRequestHandler(SearchFilesSchema, async (request, extra) => {
        try {
            const results = [];
            for (const dir of serverState.allowedDirectories) {
                if (await isPathAllowed(dir)) {
                    const matches = await searchDirectory(dir, request.params.pattern, request.params.excludePatterns);
                    results.push(...matches);
                }
            }
            return { matches: results };
        }
        catch (error) {
            log("error", `Failed to search files: ${error.message}`);
            throw new Error(`Failed to search files: ${error.message}`);
        }
    });
    server.setRequestHandler(ListAllowedDirectoriesSchema, async (request, extra) => {
        return { directories: Array.from(serverState.allowedDirectories) };
    });
    return server;
}
/**
 * Handles incoming MCP requests
 * This is the main message handler that processes all client requests
 * @param message - The incoming message to process
 */
async function handleMessage(message) {
    try {
        if (!message || typeof message !== "object") {
            log("error", "Invalid message received", message);
            return;
        }
        serverState.activeConnections++;
        if ("method" in message && "id" in message) {
            const request = message;
            log("debug", `Received request: ${request.method}`, request);
            // Handle ping requests
            if (request.method === "ping") {
                await handlePing(request);
                return;
            }
            // Handle tool requests
            const tools = SERVER_CAPABILITIES.tools;
            if (!tools || !(request.method in tools)) {
                await sendError(request.id, -32601, `Method not found: ${request.method}`);
                return;
            }
            try {
                // Validate allowed directories before processing
                if (request.params &&
                    typeof request.params === "object" &&
                    "path" in request.params) {
                    const filePath = request.params.path;
                    if (!(await isPathAllowed(filePath))) {
                        throw new Error("Path not in allowed directories");
                    }
                }
                // Handle the request based on the method
                switch (request.method) {
                    case "read_file":
                        await handleReadFile(request);
                        break;
                    case "list_dir":
                        await handleListDir(request);
                        break;
                    case "write_file":
                        await handleWriteFile(request);
                        break;
                    case "move_file":
                        await handleMoveFile(request);
                        break;
                    case "search_files":
                        await handleSearchFiles(request);
                        break;
                    case "list_allowed_directories":
                        await handleListAllowedDirectories(request);
                        break;
                    default:
                        await sendError(request.id, -32601, `Method not found: ${request.method}`);
                }
            }
            catch (error) {
                log("error", `Error handling request: ${error.message}`, {
                    method: request.method,
                    params: request.params,
                    error,
                });
                await sendError(request.id, -32000, error.message);
            }
        }
    }
    catch (error) {
        log("error", "Error processing message:", error);
    }
    finally {
        serverState.activeConnections--;
    }
}
// Update handler functions with typed parameters
async function handleReadFile(request) {
    const validatedPath = await validatePath(request.params.path);
    log("debug", `Reading file: ${validatedPath}`);
    const stats = await fs.promises.stat(validatedPath);
    if (!stats.isFile()) {
        throw new Error("Path is not a file");
    }
    const content = await fs.promises.readFile(validatedPath, "utf8");
    log("debug", `Successfully read file: ${validatedPath}`, {
        size: stats.size,
    });
    await sendResult(request.id, { content, size: stats.size });
}
async function handleListDir(request) {
    const validatedPath = await validatePath(request.params.path);
    log("debug", `Listing directory: ${validatedPath}`);
    const stats = await fs.promises.stat(validatedPath);
    if (!stats.isDirectory()) {
        throw new Error("Path is not a directory");
    }
    const files = await fs.promises.readdir(validatedPath, {
        withFileTypes: true,
    });
    const items = files.map((file) => ({
        name: file.name,
        type: file.isDirectory() ? "directory" : "file",
        size: file.isFile()
            ? fs.statSync(path.join(validatedPath, file.name)).size
            : null,
        lastModified: fs
            .statSync(path.join(validatedPath, file.name))
            .mtime.toISOString(),
    }));
    log("debug", `Successfully listed directory: ${validatedPath}`, {
        itemCount: items.length,
    });
    await sendResult(request.id, { items, totalItems: items.length });
}
async function handleWriteFile(request) {
    const validatedPath = await validatePath(request.params.path, false);
    if (!(await isPathAllowed(validatedPath))) {
        throw new Error("Path not in allowed directories");
    }
    log("debug", `Writing file: ${validatedPath}`);
    await fs.promises.writeFile(validatedPath, request.params.content, "utf8");
    const stats = await fs.promises.stat(validatedPath);
    await sendResult(request.id, { size: stats.size });
}
async function handleMoveFile(request) {
    const sourcePath = await validatePath(request.params.source);
    const destPath = await validatePath(request.params.destination, false);
    if (!(await isPathAllowed(sourcePath)) || !(await isPathAllowed(destPath))) {
        throw new Error("Path not in allowed directories");
    }
    log("debug", `Moving file: ${sourcePath} to ${destPath}`);
    await fs.promises.rename(sourcePath, destPath);
    await sendResult(request.id, { success: true });
}
async function handleSearchFiles(request) {
    const results = [];
    for (const dir of serverState.allowedDirectories) {
        if (await isPathAllowed(dir)) {
            const matches = await searchDirectory(dir, request.params.pattern, request.params.excludePatterns);
            results.push(...matches);
        }
    }
    await sendResult(request.id, { matches: results });
}
async function handleListAllowedDirectories(request) {
    await sendResult(request.id, {
        directories: Array.from(serverState.allowedDirectories),
    });
}
// Initialize allowed directories from environment
if (process.env.ALLOWED_DIRECTORIES) {
    serverState.allowedDirectories = new Set(process.env.ALLOWED_DIRECTORIES.split(","));
}
// Increase initialization timeout for tests
const INIT_TIMEOUT = process.env.NODE_ENV === "test" ? 120000 : 5000;
/**
 * Initializes the MCP server and sets up all handlers
 */
async function initializeServer() {
    try {
        // Create and initialize server
        const server = await createServer();
        // Set initialization timeout
        serverState.initializationTimeout = setTimeout(() => {
            log("error", "Server initialization timed out");
            shutdown("initialization_timeout");
        }, INIT_TIMEOUT);
        // Connect server to transport
        await server.connect(transport);
        log("info", "Server successfully connected to transport");
        // Clear initialization timeout
        if (serverState.initializationTimeout) {
            clearTimeout(serverState.initializationTimeout);
            serverState.initializationTimeout = null;
        }
        // Set up error handlers
        transport.onerror = (error) => {
            log("error", "Transport error occurred:", error);
            log("error", "Error stack:", error.stack);
        };
        // Set up signal handlers
        process.on("SIGINT", () => shutdown("SIGINT"));
        process.on("SIGTERM", () => shutdown("SIGTERM"));
        process.on("SIGHUP", () => shutdown("SIGHUP"));
        process.on("disconnect", () => shutdown("disconnect"));
        // Set up error handlers
        process.on("uncaughtException", (error) => {
            log("error", "Uncaught exception:", error);
            shutdown("uncaughtException");
        });
        process.on("unhandledRejection", (reason, promise) => {
            log("error", "Unhandled rejection:", { reason, promise });
            shutdown("unhandledRejection");
        });
        // Memory monitoring - only in production
        if (process.env.NODE_ENV !== "test") {
            setInterval(() => {
                const memoryUsage = process.memoryUsage();
                log("debug", "Memory usage stats", memoryUsage);
            }, 300000);
        }
    }
    catch (error) {
        log("error", "Failed to start server:", error);
        if (serverState.initializationTimeout) {
            clearTimeout(serverState.initializationTimeout);
        }
        process.exit(1);
    }
}
// Start the server
initializeServer().catch((error) => {
    log("error", "Failed to initialize server:", error);
    process.exit(1);
});
/**
 * Graceful shutdown handler with improved logging
 */
async function shutdown(signal) {
    if (serverState.isShuttingDown) {
        log("warn", "Shutdown already in progress, ignoring signal", { signal });
        return;
    }
    serverState.isShuttingDown = true;
    log("info", "Initiating graceful shutdown", {
        signal,
        activeConnections: serverState.activeConnections,
    });
    // Set a shorter timeout for Windows to avoid hanging
    const maxWait = process.platform === "win32" ? 1000 : 2000;
    const start = Date.now();
    // Clear any pending initialization timeout
    if (serverState.initializationTimeout) {
        clearTimeout(serverState.initializationTimeout);
        serverState.initializationTimeout = null;
    }
    // Wait for active connections to complete with timeout
    while (serverState.activeConnections > 0 && Date.now() - start < maxWait) {
        log("info", "Waiting for connections to complete", {
            activeConnections: serverState.activeConnections,
            timeWaiting: Date.now() - start,
            maxWait,
        });
        await new Promise((resolve) => setTimeout(resolve, 100));
    }
    if (serverState.activeConnections > 0) {
        log("warn", "Forcing shutdown with active connections", {
            activeConnections: serverState.activeConnections,
        });
    }
    // Clean up resources
    if (transport) {
        transport.onmessage = () => { };
        transport.onerror = () => { };
        try {
            await transport.close();
        }
        catch (error) {
            log("error", "Error closing transport:", error);
        }
    }
    log("info", "Server shutdown complete", {
        uptime: process.uptime(),
        startTime: serverState.serverStartTime?.toISOString(),
        endTime: new Date().toISOString(),
    });
    // Force exit after cleanup
    process.exit(0);
}
