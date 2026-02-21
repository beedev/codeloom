/**
 * FileTree Component
 *
 * Recursive collapsible file tree built from FileTreeNode.
 * Directories toggle open/closed; files emit onSelect.
 */

import { useState, useCallback } from 'react';
import { ChevronRight, ChevronDown, File, Folder, FolderOpen } from 'lucide-react';
import type { FileTreeNode } from '../types/index.ts';

interface FileTreeProps {
  tree: FileTreeNode;
  selectedFileId?: string;
  onSelectFile: (fileId: string, filePath: string) => void;
}

export function FileTree({ tree, selectedFileId, onSelectFile }: FileTreeProps) {
  return (
    <div className="text-sm">
      {tree.children ? (
        tree.children.map((child) => (
          <TreeNode
            key={child.name}
            node={child}
            depth={0}
            selectedFileId={selectedFileId}
            onSelectFile={onSelectFile}
          />
        ))
      ) : (
        <TreeNode
          node={tree}
          depth={0}
          selectedFileId={selectedFileId}
          onSelectFile={onSelectFile}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// TreeNode (recursive)
// ---------------------------------------------------------------------------

interface TreeNodeProps {
  node: FileTreeNode;
  depth: number;
  selectedFileId?: string;
  onSelectFile: (fileId: string, filePath: string) => void;
}

function TreeNode({ node, depth, selectedFileId, onSelectFile }: TreeNodeProps) {
  const [isOpen, setIsOpen] = useState(depth < 2);
  const isDirectory = node.type === 'directory';
  const isSelected = node.file_id === selectedFileId;

  const handleClick = useCallback(() => {
    if (isDirectory) {
      setIsOpen((prev) => !prev);
    } else if (node.file_id && node.file_path) {
      onSelectFile(node.file_id, node.file_path);
    }
  }, [isDirectory, node.file_id, node.file_path, onSelectFile]);

  return (
    <div>
      <button
        onClick={handleClick}
        className={`flex w-full items-center gap-1.5 rounded px-1 py-1 text-left transition-colors ${
          isSelected
            ? 'bg-glow/10 text-glow'
            : 'text-text-muted hover:bg-void-surface hover:text-text'
        }`}
        style={{ paddingLeft: `${depth * 16 + 4}px` }}
      >
        {/* Chevron for directories */}
        {isDirectory ? (
          isOpen ? (
            <ChevronDown className="h-3.5 w-3.5 shrink-0 text-text-dim" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 shrink-0 text-text-dim" />
          )
        ) : (
          <span className="w-3.5 shrink-0" />
        )}

        {/* Icon */}
        {isDirectory ? (
          isOpen ? (
            <FolderOpen className="h-4 w-4 shrink-0 text-glow" />
          ) : (
            <Folder className="h-4 w-4 shrink-0 text-text-dim" />
          )
        ) : (
          <File className="h-4 w-4 shrink-0 text-text-dim" />
        )}

        {/* Name */}
        <span className="truncate text-xs">{node.name}</span>

        {/* Line count for files */}
        {!isDirectory && node.line_count != null && (
          <span className="ml-auto shrink-0 text-[10px] text-text-dim">
            {node.line_count}L
          </span>
        )}
      </button>

      {/* Children */}
      {isDirectory && isOpen && node.children && (
        <div>
          {node.children.map((child) => (
            <TreeNode
              key={child.name}
              node={child}
              depth={depth + 1}
              selectedFileId={selectedFileId}
              onSelectFile={onSelectFile}
            />
          ))}
        </div>
      )}
    </div>
  );
}
