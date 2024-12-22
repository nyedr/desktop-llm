import * as path from "path";
import * as fs from "fs/promises";
import * as os from "os";
import { getFileStats, searchFiles, applyFileEdits } from "../file-operations";
describe("File Operations", () => {
    let testDir;
    beforeEach(async () => {
        testDir = path.join(os.tmpdir(), "mcp-fs-test-ops");
        await fs.mkdir(testDir, { recursive: true });
    });
    afterEach(async () => {
        await fs.rm(testDir, { recursive: true, force: true });
    });
    describe("getFileStats", () => {
        it("should return correct file information", async () => {
            const testFile = path.join(testDir, "test.txt");
            await fs.writeFile(testFile, "test content");
            const stats = await getFileStats(testFile);
            expect(stats.type).toBe("file");
            expect(stats.size).toBe(12); // 'test content' length
            expect(stats.permissions).toMatch(/^[0-7]{3}$/);
            expect(stats.created).toBeTruthy();
            expect(stats.modified).toBeTruthy();
            expect(stats.accessed).toBeTruthy();
            const now = Date.now();
            const fiveMinutesAgo = now - 5 * 60 * 1000;
            expect(new Date(stats.created).getTime()).toBeGreaterThan(fiveMinutesAgo);
            expect(new Date(stats.modified).getTime()).toBeGreaterThan(fiveMinutesAgo);
            expect(new Date(stats.accessed).getTime()).toBeGreaterThan(fiveMinutesAgo);
        });
        it("should return correct directory information", async () => {
            const subdir = path.join(testDir, "subdir");
            await fs.mkdir(subdir);
            const stats = await getFileStats(subdir);
            expect(stats.type).toBe("directory");
            expect(stats.permissions).toMatch(/^[0-7]{3}$/);
            expect(stats.created).toBeTruthy();
            expect(stats.modified).toBeTruthy();
            expect(stats.accessed).toBeTruthy();
            const now = Date.now();
            const fiveMinutesAgo = now - 5 * 60 * 1000;
            expect(new Date(stats.created).getTime()).toBeGreaterThan(fiveMinutesAgo);
            expect(new Date(stats.modified).getTime()).toBeGreaterThan(fiveMinutesAgo);
            expect(new Date(stats.accessed).getTime()).toBeGreaterThan(fiveMinutesAgo);
        });
    });
    describe("searchFiles", () => {
        beforeEach(async () => {
            await fs.writeFile(path.join(testDir, "file1.txt"), "content1");
            await fs.writeFile(path.join(testDir, "file2.js"), "content2");
            const subdir = path.join(testDir, "subdir");
            await fs.mkdir(subdir);
            await fs.writeFile(path.join(subdir, "file3.txt"), "content3");
        });
        it("should find files matching pattern", async () => {
            const results = await searchFiles(testDir, "*.txt");
            expect(results).toHaveLength(2);
            expect(results.map((r) => path.basename(r))).toEqual(expect.arrayContaining(["file1.txt", "file3.txt"]));
        });
        it("should respect exclude patterns", async () => {
            const results = await searchFiles(testDir, "*.*", ["*.js"]);
            expect(results).toHaveLength(2);
            expect(results.map((r) => path.basename(r))).toEqual(expect.arrayContaining(["file1.txt", "file3.txt"]));
        });
    });
    describe("applyFileEdits", () => {
        let testFile;
        beforeEach(async () => {
            testFile = path.join(testDir, "edit-test.txt");
        });
        it("should apply simple text replacement", async () => {
            await fs.writeFile(testFile, "line1\nline2\nline3");
            const edits = [
                {
                    oldText: "line2",
                    newText: "replaced",
                    startLine: 2,
                    endLine: 2,
                },
            ];
            const diff = await applyFileEdits(testFile, edits);
            const content = await fs.readFile(testFile, "utf8");
            expect(content).toBe("line1\nreplaced\nline3");
            expect(diff).toContain("@@ -1,3 +1,3 @@");
            expect(diff).toContain("-line2");
            expect(diff).toContain("+replaced");
        });
        it("should handle whitespace-only differences", async () => {
            await fs.writeFile(testFile, "line1\nline2  \nline3");
            const edits = [
                {
                    oldText: "line2  ",
                    newText: "line2",
                    startLine: 2,
                    endLine: 2,
                },
            ];
            const diff = await applyFileEdits(testFile, edits);
            const content = await fs.readFile(testFile, "utf8");
            expect(content).toBe("line1\nline2\nline3");
            expect(diff).toContain("@@ -1,3 +1,3 @@");
            expect(diff).toContain("-line2  ");
            expect(diff).toContain("+line2");
        });
        it("should handle multiple edits", async () => {
            await fs.writeFile(testFile, "line1\nline2\nline3\nline4");
            const edits = [
                {
                    oldText: "line2",
                    newText: "replaced2",
                    startLine: 2,
                    endLine: 2,
                },
                {
                    oldText: "line4",
                    newText: "replaced4",
                    startLine: 4,
                    endLine: 4,
                },
            ];
            const diff = await applyFileEdits(testFile, edits);
            const content = await fs.readFile(testFile, "utf8");
            expect(content).toBe("line1\nreplaced2\nline3\nreplaced4");
            expect(diff).toContain("@@ -1,4 +1,4 @@");
            expect(diff).toContain("-line2");
            expect(diff).toContain("+replaced2");
            expect(diff).toContain("-line4");
            expect(diff).toContain("+replaced4");
        });
        it("should escape backticks in diff output", async () => {
            await fs.writeFile(testFile, "line with ``` backticks");
            const edits = [
                {
                    oldText: "line with ``` backticks",
                    newText: "replaced",
                    startLine: 1,
                    endLine: 1,
                },
            ];
            const diff = await applyFileEdits(testFile, edits);
            const content = await fs.readFile(testFile, "utf8");
            expect(content).toBe("replaced");
            expect(diff).toContain("-line with ``` backticks");
            expect(diff).toContain("+replaced");
        });
    });
});
