import fs from "fs/promises";
import path from "path";
import os from "os";

export function normalizePath(p: string): string {
  return path.normalize(p).toLowerCase();
}

export function expandHome(filepath: string): string {
  if (filepath.startsWith("~/") || filepath === "~") {
    return path.join(os.homedir(), filepath.slice(1));
  }
  return filepath;
}

async function getRealPath(p: string): Promise<string> {
  try {
    return await fs.realpath(p);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return p;
    }
    throw error;
  }
}

export async function validatePath(
  filePath: string,
  allowedDirectories: string[]
): Promise<string> {
  const absolute = path.resolve(filePath);
  const normalizedPath = path.normalize(absolute);

  // Normalize all allowed directories
  const normalizedAllowedDirs = allowedDirectories.map((dir) =>
    path.normalize(path.resolve(dir))
  );

  // For symlinks, validate the target first
  try {
    const stats = await fs.lstat(normalizedPath);
    if (stats.isSymbolicLink()) {
      const target = await fs.readlink(normalizedPath);
      const absoluteTarget = path.resolve(path.dirname(normalizedPath), target);
      const normalizedTarget = path.normalize(absoluteTarget);

      const isTargetAllowed = normalizedAllowedDirs.some((dir) =>
        normalizedTarget.startsWith(dir)
      );

      if (!isTargetAllowed) {
        throw new Error(
          `Access denied - symlink target outside allowed directories: ${normalizedTarget} not in ${normalizedAllowedDirs.join(
            ", "
          )}`
        );
      }
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
      throw error;
    }
  }

  // Check if the path is within any of the allowed directories
  const isAllowed = normalizedAllowedDirs.some((dir) =>
    normalizedPath.startsWith(dir)
  );

  if (!isAllowed) {
    throw new Error(
      `Access denied - path outside allowed directories: ${normalizedPath} not in ${normalizedAllowedDirs.join(
        ", "
      )}`
    );
  }

  // If the file doesn't exist, validate its parent directory
  try {
    await fs.access(normalizedPath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      const parentDir = path.dirname(normalizedPath);
      const parentStats = await fs.stat(parentDir);
      if (!parentStats.isDirectory()) {
        throw new Error("Parent path is not a directory");
      }
    } else {
      throw error;
    }
  }

  return normalizedPath;
}
