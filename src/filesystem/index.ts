import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  CallToolRequest,
  JSONRPCRequest,
  JSONRPCResponse,
  JSONRPCError,
} from "@modelcontextprotocol/sdk/types.js";
import type {
  Request,
  Notification,
  Result,
  JSONRPCMessage,
} from "@modelcontextprotocol/sdk/types.js";

/**
 * Debug logging function that writes to stderr to avoid interfering with stdio transport
 * @param level - Log level (debug, info, warn, error)
 * @param message - Message to log
 * @param data - Optional data to log
 */
function log(level: string, message: string, data?: any) {
  const timestamp = new Date().toISOString();
  const prefix = `[Node.js ${
    process.pid
  }] [${level.toUpperCase()}] [${timestamp}]`;
  console.error(`${prefix} ${message}`);
  if (data) {
    console.error(
      `${prefix} Data:`,
      typeof data === "string" ? data : JSON.stringify(data, null, 2)
    );
  }
}

// Log startup information
log("info", "MCP Server starting...");
log(
  "info",
  `Process info - PID: ${process.pid}, Node: ${process.version}, Platform: ${process.platform}`
);
log("debug", "Environment variables:", process.env);

// Create transport with debug logging
const transport = new StdioServerTransport();

// Track server state
let isShuttingDown = false;
let activeConnections = 0;

transport.onmessage = (message: Request | Notification | Result) => {
  try {
    activeConnections++;
    log("debug", "Received message", message);

    // Handle ping message specially
    if ("method" in message && message.method === "ping" && "id" in message) {
      log("debug", "Processing ping request");
      const response: JSONRPCResponse = {
        jsonrpc: "2.0",
        id: (message as JSONRPCRequest).id,
        result: {
          status: "ok",
          server_info: {
            pid: process.pid,
            version: process.version,
            uptime: process.uptime(),
          },
        },
      };
      log("debug", "Sending ping response", response);
      transport.send(response);
      return;
    }

    // Handle other message types
    if ("method" in message) {
      log("debug", `Received method call: ${message.method}`);
    } else if ("result" in message) {
      log("debug", "Received result", message.result);
    } else if ("error" in message) {
      log("warn", "Received error", message.error);
    }
  } catch (error) {
    log("error", "Error handling message:", error);
    // Try to send error response if possible
    if ("id" in message) {
      const errorResponse: JSONRPCError = {
        jsonrpc: "2.0",
        id: (message as JSONRPCRequest).id,
        error: {
          code: -32603,
          message: `Internal error: ${error}`,
          data: { stack: error instanceof Error ? error.stack : undefined },
        },
      };
      transport.send(errorResponse);
    }
  } finally {
    activeConnections--;
  }
};

transport.onerror = (error: Error) => {
  log("error", "Transport error:", error);
  log("error", "Error stack:", error.stack);
};

// Create server with capabilities
const server = new Server(
  {
    name: "secure-filesystem-server",
    version: "0.2.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Add initialization handler with error handling
server.oninitialized = () => {
  try {
    log("info", "Server initialization completed");
  } catch (error) {
    log("error", "Error in initialization handler:", error);
  }
};

/**
 * Graceful shutdown handler
 */
async function shutdown(signal: string) {
  if (isShuttingDown) {
    log("warn", "Shutdown already in progress");
    return;
  }

  isShuttingDown = true;
  log("info", `Initiating graceful shutdown (signal: ${signal})`);

  // Wait for active connections to complete (with timeout)
  const maxWait = 5000;
  const start = Date.now();

  while (activeConnections > 0 && Date.now() - start < maxWait) {
    log(
      "info",
      `Waiting for ${activeConnections} active connections to complete...`
    );
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  if (activeConnections > 0) {
    log(
      "warn",
      `Forcing shutdown with ${activeConnections} active connections`
    );
  }

  process.exit(0);
}

// Connect server to transport with error handling
try {
  await server.connect(transport);
  log("info", "Server successfully connected to transport");

  // Keep the process running
  process.stdin.resume();

  // Handle various signals for graceful shutdown
  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));
  process.on("SIGHUP", () => shutdown("SIGHUP"));

  // Handle uncaught errors
  process.on("uncaughtException", (error) => {
    log("error", "Uncaught exception:", error);
    shutdown("uncaughtException");
  });

  process.on("unhandledRejection", (reason, promise) => {
    log("error", "Unhandled rejection:", { reason, promise });
    shutdown("unhandledRejection");
  });

  // Monitor memory usage
  setInterval(() => {
    const memoryUsage = process.memoryUsage();
    log("debug", "Memory usage:", memoryUsage);
  }, 30000);
} catch (error) {
  log("error", "Failed to start server:", error);
  process.exit(1);
}
