import * as path from "path";
import * as fs from "fs/promises";
import glob from "glob";
import { promisify } from "util";
import { createTwoFilesPatch } from "diff";
const globAsync = promisify(glob);
export async function getFileStats(filePath) {
    const stats = await fs.stat(filePath);
    return {
        type: stats.isDirectory() ? "directory" : "file",
        size: stats.size,
        created: stats.birthtime.toISOString(),
        modified: stats.mtime.toISOString(),
        accessed: stats.atime.toISOString(),
        permissions: stats.mode.toString(8).slice(-3),
    };
}
export async function searchFiles(directory, pattern, excludePatterns = []) {
    const options = {
        cwd: directory,
        ignore: excludePatterns,
        absolute: true,
        dot: true,
        nodir: true,
        matchBase: true,
    };
    const fullPattern = pattern.includes("/") ? pattern : `**/${pattern}`;
    const files = await globAsync(fullPattern, options);
    return files.map((file) => path.normalize(file));
}
function normalizeLineEndings(text) {
    return text.replace(/\r\n/g, "\n");
}
function createUnifiedDiff(originalContent, newContent, filepath = "file") {
    // Ensure consistent line endings for diff
    const normalizedOriginal = normalizeLineEndings(originalContent);
    const normalizedNew = normalizeLineEndings(newContent);
    return createTwoFilesPatch(filepath, filepath, normalizedOriginal, normalizedNew, "original", "modified");
}
export async function applyFileEdits(filePath, edits, dryRun = false) {
    if (edits.length === 0) {
        return ""; // Return empty string for no edits
    }
    // Read file content and normalize line endings
    const content = normalizeLineEndings(await fs.readFile(filePath, "utf-8"));
    // Apply edits sequentially
    let modifiedContent = content;
    for (const edit of edits) {
        const normalizedOld = normalizeLineEndings(edit.oldText);
        const normalizedNew = normalizeLineEndings(edit.newText);
        // If exact match exists, use it
        if (modifiedContent.includes(normalizedOld)) {
            modifiedContent = modifiedContent.replace(normalizedOld, normalizedNew);
            continue;
        }
        // Otherwise, try line-by-line matching with flexibility for whitespace
        const oldLines = normalizedOld.split("\n");
        const contentLines = modifiedContent.split("\n");
        let matchFound = false;
        for (let i = 0; i <= contentLines.length - oldLines.length; i++) {
            const potentialMatch = contentLines.slice(i, i + oldLines.length);
            // Compare lines with normalized whitespace
            const isMatch = oldLines.every((oldLine, j) => {
                const contentLine = potentialMatch[j];
                return oldLine.trim() === contentLine?.trim();
            });
            if (isMatch) {
                // Preserve original indentation of first line
                const originalIndent = contentLines[i].match(/^\s*/)?.[0] || "";
                const newLines = normalizedNew.split("\n").map((line, j) => {
                    if (j === 0)
                        return originalIndent + line.trimStart();
                    // For subsequent lines, try to preserve relative indentation
                    const oldIndent = oldLines[j]?.match(/^\s*/)?.[0] || "";
                    const newIndent = line.match(/^\s*/)?.[0] || "";
                    if (oldIndent && newIndent) {
                        const relativeIndent = newIndent.length - oldIndent.length;
                        return (originalIndent +
                            " ".repeat(Math.max(0, relativeIndent)) +
                            line.trimStart());
                    }
                    return line;
                });
                contentLines.splice(i, oldLines.length, ...newLines);
                modifiedContent = contentLines.join("\n");
                matchFound = true;
                break;
            }
        }
        if (!matchFound) {
            throw new Error(`Could not find exact match for edit:\n${edit.oldText}`);
        }
    }
    // Create unified diff
    const diff = createUnifiedDiff(content, modifiedContent, filePath);
    if (!dryRun) {
        await fs.writeFile(filePath, modifiedContent, "utf-8");
    }
    return diff;
}
