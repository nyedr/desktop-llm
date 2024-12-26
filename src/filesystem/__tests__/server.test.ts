/**
 * @jest-environment node
 */

import { jest } from "@jest/globals";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  JSONRPCRequest,
  InitializeRequest,
  InitializedNotification,
  LATEST_PROTOCOL_VERSION,
} from "@modelcontextprotocol/sdk/types.js";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// Mock the transport with proper types
const mockSend = jest.fn();
const mockOnMessage = jest.fn();

// Define types for filesystem mocks
interface FSStats {
  isFile: () => boolean;
  isDirectory: () => boolean;
  size: number;
  mtime: Date;
}

interface DirEnt {
  name: string;
  isFile: () => boolean;
  isDirectory: () => boolean;
}

// Create mock functions with correct function signatures
const mockReadFile = jest.fn(async (...args: any[]) => "test content");
const mockWriteFile = jest.fn(async (...args: any[]) => undefined);
const mockStat = jest.fn(
  async (...args: any[]) =>
    ({
      isFile: () => true,
      isDirectory: () => false,
      size: 100,
      mtime: new Date(),
    } as FSStats)
);

const mockDirEnt = {
  name: "test.txt",
  isFile: () => true,
  isDirectory: () => false,
} as DirEnt;

const mockReaddir = jest.fn(async (...args: any[]) => [mockDirEnt]);
const mockRename = jest.fn(async (...args: any[]) => undefined);
const mockStatSync = jest.fn(
  (...args: any[]) =>
    ({
      isFile: () => true,
      isDirectory: () => false,
      size: 100,
      mtime: new Date(),
    } as FSStats)
);

// Mock the fs module
jest.unstable_mockModule("fs", () => ({
  promises: {
    readFile: mockReadFile,
    writeFile: mockWriteFile,
    stat: mockStat,
    readdir: mockReaddir,
    rename: mockRename,
  },
  statSync: mockStatSync,
}));

describe("MCP Server", () => {
  const testDir = path.join(os.tmpdir(), "mcp-fs-test-server");
  let requestId = 1;

  beforeAll(() => {
    // Set up test environment
    process.env.ALLOWED_DIRECTORIES = testDir;
    process.env.NODE_ENV = "test";
  });

  afterAll(() => {
    delete process.env.ALLOWED_DIRECTORIES;
    delete process.env.NODE_ENV;
  });

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();

    // Set up filesystem mocks with default behaviors
    mockStat.mockResolvedValue({
      isFile: () => true,
      isDirectory: () => false,
      size: 100,
      mtime: new Date(),
    } as FSStats);
    mockReadFile.mockResolvedValue("test content");
    mockWriteFile.mockResolvedValue(undefined);
    mockReaddir.mockResolvedValue([]);
    mockRename.mockResolvedValue(undefined);
    mockStatSync.mockReturnValue({
      isFile: () => true,
      isDirectory: () => false,
      size: 100,
      mtime: new Date(),
    } as FSStats);
  });

  describe("Server Initialization", () => {
    it("should initialize with correct capabilities", async () => {
      const initRequest = {
        jsonrpc: "2.0" as const,
        id: requestId++,
        method: "initialize",
        params: {
          protocolVersion: LATEST_PROTOCOL_VERSION,
          clientInfo: {
            name: "test-client",
            version: "1.0.0",
          },
          capabilities: {
            tools: true,
          },
        },
      } as unknown as InitializeRequest;

      // Send initialization request
      mockOnMessage(initRequest);

      // Verify initialization response
      expect(mockSend).toHaveBeenCalledWith(
        expect.objectContaining({
          jsonrpc: "2.0",
          id: 1,
          result: expect.objectContaining({
            protocolVersion: expect.any(String),
            capabilities: expect.objectContaining({
              tools: expect.any(Object),
              logging: expect.any(Object),
            }),
            serverInfo: expect.objectContaining({
              name: "secure-filesystem-server",
              version: expect.any(String),
            }),
          }),
        })
      );

      // Send initialized notification
      mockOnMessage({
        jsonrpc: "2.0" as const,
        method: "notifications/initialized",
      } as unknown as InitializedNotification);
    });
  });

  describe("Filesystem Operations", () => {
    beforeEach(async () => {
      const initRequest = {
        jsonrpc: "2.0" as const,
        id: requestId++,
        method: "initialize",
        params: {
          protocolVersion: LATEST_PROTOCOL_VERSION,
          clientInfo: {
            name: "test-client",
            version: "1.0.0",
          },
          capabilities: {
            tools: true,
          },
        },
      } as unknown as InitializeRequest;

      mockOnMessage(initRequest);
      mockOnMessage({
        jsonrpc: "2.0" as const,
        method: "notifications/initialized",
      } as unknown as InitializedNotification);
    });

    describe("read_file Tool", () => {
      it("should successfully read a file", async () => {
        const request = {
          jsonrpc: "2.0" as const,
          id: requestId++,
          method: "read_file",
          params: {
            path: path.join(testDir, "test.txt"),
          },
        } as unknown as JSONRPCRequest;

        mockOnMessage(request);

        expect(mockReadFile).toHaveBeenCalledWith(
          path.join(testDir, "test.txt"),
          "utf8"
        );
        expect(mockSend).toHaveBeenCalledWith(
          expect.objectContaining({
            jsonrpc: "2.0",
            id: request.id,
            result: expect.objectContaining({
              content: "test content",
              size: expect.any(Number),
            }),
          })
        );
      });

      it("should handle non-existent files", async () => {
        mockStat.mockRejectedValueOnce(new Error("File not found"));

        const request = {
          jsonrpc: "2.0" as const,
          id: requestId++,
          method: "read_file",
          params: {
            path: path.join(testDir, "nonexistent.txt"),
          },
        } as unknown as JSONRPCRequest;

        mockOnMessage(request);

        expect(mockSend).toHaveBeenCalledWith(
          expect.objectContaining({
            jsonrpc: "2.0",
            id: request.id,
            error: expect.objectContaining({
              code: -32000,
              message: expect.stringContaining("File not found"),
            }),
          })
        );
      });
    });

    describe("list_dir Tool", () => {
      it("should successfully list directory contents", async () => {
        mockStat.mockResolvedValueOnce({
          isFile: () => false,
          isDirectory: () => true,
          size: 0,
          mtime: new Date(),
        } as FSStats);
        mockReaddir.mockResolvedValueOnce([
          { name: "file1.txt", isFile: () => true, isDirectory: () => false },
          { name: "dir1", isFile: () => false, isDirectory: () => true },
        ]);

        const request = {
          jsonrpc: "2.0" as const,
          id: requestId++,
          method: "list_dir",
          params: {
            path: testDir,
          },
        } as unknown as JSONRPCRequest;

        mockOnMessage(request);

        expect(mockReaddir).toHaveBeenCalledWith(testDir, {
          withFileTypes: true,
        });
        expect(mockSend).toHaveBeenCalledWith(
          expect.objectContaining({
            jsonrpc: "2.0",
            id: request.id,
            result: expect.objectContaining({
              items: expect.arrayContaining([
                expect.objectContaining({
                  name: "file1.txt",
                  type: "file",
                }),
                expect.objectContaining({
                  name: "dir1",
                  type: "directory",
                }),
              ]),
            }),
          })
        );
      });
    });

    describe("write_file Tool", () => {
      it("should successfully write to a file", async () => {
        const request = {
          jsonrpc: "2.0" as const,
          id: requestId++,
          method: "write_file",
          params: {
            path: path.join(testDir, "new.txt"),
            content: "new content",
          },
        } as unknown as JSONRPCRequest;

        mockOnMessage(request);

        expect(mockWriteFile).toHaveBeenCalledWith(
          path.join(testDir, "new.txt"),
          "new content",
          "utf8"
        );
        expect(mockSend).toHaveBeenCalledWith(
          expect.objectContaining({
            jsonrpc: "2.0",
            id: request.id,
            result: expect.objectContaining({
              size: expect.any(Number),
            }),
          })
        );
      });
    });

    describe("Error Handling", () => {
      it("should handle invalid paths", async () => {
        const request = {
          jsonrpc: "2.0" as const,
          id: requestId++,
          method: "read_file",
          params: {
            path: path.join(os.tmpdir(), "outside", "test.txt"),
          },
        } as unknown as JSONRPCRequest;

        mockOnMessage(request);

        expect(mockSend).toHaveBeenCalledWith(
          expect.objectContaining({
            jsonrpc: "2.0",
            id: request.id,
            error: expect.objectContaining({
              code: -32000,
              message: expect.stringContaining(
                "Path not in allowed directories"
              ),
            }),
          })
        );
      });

      it("should handle unknown methods", async () => {
        const request = {
          jsonrpc: "2.0" as const,
          id: requestId++,
          method: "unknown_method",
          params: {},
        } as unknown as JSONRPCRequest;

        mockOnMessage(request);

        expect(mockSend).toHaveBeenCalledWith(
          expect.objectContaining({
            jsonrpc: "2.0",
            id: request.id,
            error: expect.objectContaining({
              code: -32601,
              message: expect.stringContaining("Method not found"),
            }),
          })
        );
      });
    });
  });
});
