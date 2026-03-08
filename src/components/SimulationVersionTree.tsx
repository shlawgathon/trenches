"use client";

import { GitBranchPlus, RotateCcw, Undo2 } from "lucide-react";

export type SimulationBranch = {
  id: string;
  label: string;
  forkTurn: number;
  parentId: string | null;
};

type SimulationVersionTreeProps = {
  branches: SimulationBranch[];
  activeBranchId: string;
  currentTurn: number;
  onCreateBranch: () => void;
  onSelectBranch: (branchId: string, turn: number) => void;
  onRewindToTurn: (turn: number) => void;
};

export function SimulationVersionTree({
  branches,
  activeBranchId,
  currentTurn,
  onCreateBranch,
  onSelectBranch,
  onRewindToTurn,
}: SimulationVersionTreeProps) {
  return (
    <div className="border border-border/30 bg-card/55 p-3 backdrop-blur-xl">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-foreground/80">Simulation Version Tree</span>
        <button
          onClick={onCreateBranch}
          className="inline-flex items-center gap-1 border border-border/40 px-1.5 py-1 text-[9px] font-mono uppercase tracking-wider text-muted-foreground hover:text-foreground"
        >
          <GitBranchPlus className="h-3 w-3" />
          Branch @T{currentTurn}
        </button>
      </div>

      <div className="max-h-48 space-y-1 overflow-y-auto pr-1">
        {branches.map((branch) => {
          const depth = getDepth(branches, branch.id);
          const isActive = branch.id === activeBranchId;
          return (
            <div
              key={branch.id}
              className="flex items-center gap-1"
              style={{ paddingLeft: `${depth * 14}px` }}
            >
              <button
                onClick={() => onSelectBranch(branch.id, branch.forkTurn)}
                className={`flex-1 border px-2 py-1 text-left text-[9px] font-mono uppercase tracking-wider ${
                  isActive
                    ? "border-primary/60 bg-primary/10 text-primary"
                    : "border-border/40 text-muted-foreground hover:text-foreground"
                }`}
              >
                {branch.label} · T{branch.forkTurn}
              </button>
              <button
                onClick={() => onRewindToTurn(branch.forkTurn)}
                className="border border-border/40 p-1 text-muted-foreground hover:text-foreground"
                title={`Rewind to T${branch.forkTurn}`}
              >
                <Undo2 className="h-3 w-3" />
              </button>
            </div>
          );
        })}
      </div>

      <button
        onClick={() => onRewindToTurn(0)}
        className="mt-2 inline-flex items-center gap-1 border border-border/40 px-2 py-1 text-[9px] font-mono uppercase tracking-wider text-muted-foreground hover:text-foreground"
      >
        <RotateCcw className="h-3 w-3" />
        Reverse to Root
      </button>
    </div>
  );
}

function getDepth(branches: SimulationBranch[], id: string): number {
  const lookup = new Map(branches.map((branch) => [branch.id, branch]));
  let depth = 0;
  let current = lookup.get(id);

  while (current?.parentId) {
    depth += 1;
    current = lookup.get(current.parentId);
  }

  return depth;
}
