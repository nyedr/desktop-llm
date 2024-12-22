import * as path from "path";
import * as fs from "fs/promises";
import * as os from "os";
import { validatePath } from "../path-utils";

describe("Path Utilities", () => {
  describe("validatePath", () => {
    let testDir: string;
    let testFile: string;

    beforeEach(async () => {
      testDir = path.join(os.tmpdir(), "mcp-fs-test");
      await fs.mkdir(testDir, { recursive: true });
      testFile = path.join(testDir, "test.txt");
      await fs.writeFile(testFile, "test content");
    });

    afterEach(async () => {
      await fs.rm(testDir, { recursive: true, force: true });
    });

    it("should allow paths within allowed directories", async () => {
      await expect(validatePath(testFile, [testDir])).resolves.not.toThrow();
    });

    it("should handle symlinks within allowed directories", async () => {
      const symlink = path.join(testDir, "link.txt");
      await fs.symlink(testFile, symlink);
      await expect(validatePath(symlink, [testDir])).resolves.not.toThrow();
    });

    it("should reject symlinks pointing outside allowed directories", async () => {
      const outsideFile = path.join(os.tmpdir(), "outside.txt");
      await fs.writeFile(outsideFile, "outside content");
      const symlink = path.join(testDir, "bad-link.txt");
      await fs.symlink(outsideFile, symlink);

      await expect(validatePath(symlink, [testDir])).rejects.toThrow(
        /Access denied - symlink target outside allowed directories/
      );

      await fs.unlink(outsideFile);
    });

    it("should validate parent directory for new files", async () => {
      const subdir = path.join(testDir, "subdir");
      await fs.mkdir(subdir);
      const newFile = path.join(subdir, "new-file.txt");
      await expect(validatePath(newFile, [testDir])).resolves.not.toThrow();
    });

    it("should reject new files in unauthorized directories", async () => {
      const unauthorizedDir = path.join(os.tmpdir(), "unauthorized");
      await fs.mkdir(unauthorizedDir, { recursive: true });
      const newFile = path.join(unauthorizedDir, "new-file.txt");

      await expect(validatePath(newFile, [testDir])).rejects.toThrow(
        /Access denied - path outside allowed directories/
      );

      await fs.rm(unauthorizedDir, { recursive: true });
    });
  });
});
