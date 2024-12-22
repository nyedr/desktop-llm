import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
// Log startup
console.error("[Node.js] MCP Server starting...");
console.error("[Node.js] Process ID:", process.pid);
console.error("[Node.js] Node version:", process.version);
// Create transport with debug logging
const transport = new StdioServerTransport();
transport.onmessage = (message) => {
    try {
        console.error("=== Received Message ===");
        console.error("Raw message:", message);
        console.error("Message as JSON:", JSON.stringify(message, null, 2));
        console.error("=====================");
        // Handle ping message specially
        if ("method" in message && message.method === "ping" && "id" in message) {
            const response = {
                jsonrpc: "2.0",
                id: message.id,
                result: { status: "ok" },
            };
            transport.send(response);
            return;
        }
    }
    catch (error) {
        console.error("Error handling message:", error);
    }
};
transport.onerror = (error) => {
    try {
        console.error("=== Server Error ===");
        console.error("Error:", error);
        console.error("Stack:", error.stack);
        console.error("==================");
    }
    catch (innerError) {
        console.error("Error handling error event:", innerError);
    }
};
// Create server with capabilities
const server = new Server({
    name: "secure-filesystem-server",
    version: "0.2.0",
}, {
    capabilities: {
        tools: {},
    },
});
// Add initialization handler with error handling
server.oninitialized = () => {
    try {
        console.error("[Node.js] Server initialization completed");
    }
    catch (error) {
        console.error("Error in initialization handler:", error);
    }
};
// Connect server to transport with error handling
try {
    await server.connect(transport);
    console.error("[Node.js] Secure MCP Filesystem Server running on stdio");
    // Keep the process running
    process.stdin.resume();
    // Handle interrupts gracefully
    process.on("SIGINT", () => {
        console.error("[Node.js] Received SIGINT, shutting down...");
        process.exit(0);
    });
    // Handle process errors
    process.on("uncaughtException", (error) => {
        console.error("[Node.js] Uncaught exception:", error);
    });
    process.on("unhandledRejection", (reason, promise) => {
        console.error("[Node.js] Unhandled rejection at:", promise, "reason:", reason);
    });
}
catch (error) {
    console.error("[Node.js] Failed to start server:", error);
    process.exit(1);
}
