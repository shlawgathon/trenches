"use client";

import { cn } from "@/src/lib/utils";

export type AgentVisual = {
  flag: string;
  primary: string;
  secondary: string;
};

export type MatrixMode = "absolute" | "delta";

export type TensionMatrixProps = {
  agents: string[];
  matrix: number[][];
  visuals: Record<string, AgentVisual>;
  mode: MatrixMode;
  onModeChange: (mode: MatrixMode) => void;
  className?: string;
};

function blendHex(colorA: string, colorB: string, amount: number): string {
  const a = colorA.replace("#", "");
  const b = colorB.replace("#", "");
  const mix = (i: number) => {
    const av = Number.parseInt(a.slice(i, i + 2), 16);
    const bv = Number.parseInt(b.slice(i, i + 2), 16);
    return Math.round(av * (1 - amount) + bv * amount)
      .toString(16)
      .padStart(2, "0");
  };
  return `#${mix(0)}${mix(2)}${mix(4)}`;
}

function tensionColor(value: number, rowColor: string, colColor: string): string {
  const clamped = Math.max(0, Math.min(100, Math.abs(value))) / 100;
  const baseBlend = blendHex(rowColor, colColor, 0.5);
  const low = blendHex(baseBlend, "#1e3a8a", 0.55);
  const high = blendHex(baseBlend, "#dc2626", 0.45);
  return blendHex(low, high, clamped);
}

export function TensionMatrix({ agents, matrix, visuals, mode, onModeChange, className }: TensionMatrixProps) {
  if (agents.length === 0) return null;

  return (
    <div className={cn("border border-border/30 bg-card/55 p-3 backdrop-blur-xl", className)}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-foreground/80">Inter-Agent Tension Matrix</div>
        <div className="flex items-center border border-border/40">
          <button
            onClick={() => onModeChange("absolute")}
            className={cn("px-2 py-1 text-[9px] font-mono uppercase", mode === "absolute" ? "bg-primary/20 text-primary" : "text-muted-foreground")}
          >
            Absolute
          </button>
          <button
            onClick={() => onModeChange("delta")}
            className={cn("px-2 py-1 text-[9px] font-mono uppercase", mode === "delta" ? "bg-primary/20 text-primary" : "text-muted-foreground")}
          >
            Delta
          </button>
        </div>
      </div>

      <div className="max-h-[240px] overflow-auto border border-border/20">
        <table className="w-full border-collapse text-[9px] font-mono uppercase">
          <thead>
            <tr>
              <th className="sticky top-0 left-0 z-20 bg-card px-2 py-1 text-left text-muted-foreground">Agent</th>
              {agents.map((agent) => (
                <th key={`col-${agent}`} className="sticky top-0 z-10 bg-card px-2 py-1 text-center text-muted-foreground" title={agent}>
                  <span className="mr-1">{visuals[agent]?.flag ?? "🏳️"}</span>
                  {agent.slice(0, 3)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {agents.map((rowAgent, rowIdx) => (
              <tr key={`row-${rowAgent}`}>
                <th className="sticky left-0 z-10 bg-card px-2 py-1 text-left text-muted-foreground">
                  <span className="mr-1">{visuals[rowAgent]?.flag ?? "🏳️"}</span>
                  {rowAgent.slice(0, 3)}
                </th>
                {agents.map((colAgent, colIdx) => {
                  const value = rowIdx === colIdx ? 0 : matrix[rowIdx]?.[colIdx] ?? 0;
                  const rowColor = visuals[rowAgent]?.primary ?? "#64748b";
                  const colColor = visuals[colAgent]?.secondary ?? "#94a3b8";
                  return (
                    <td
                      key={`${rowAgent}-${colAgent}`}
                      className="relative h-8 min-w-9 border border-black/20 text-center text-white/90"
                      style={{ backgroundColor: tensionColor(value, rowColor, colColor), opacity: rowIdx === colIdx ? 0.25 : 0.9 }}
                      title={`${visuals[rowAgent]?.flag ?? ""} ${rowAgent} → ${visuals[colAgent]?.flag ?? ""} ${colAgent}: ${value.toFixed(0)}`}
                    >
                      {value.toFixed(0)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex items-center justify-between text-[8px] font-mono uppercase tracking-wider text-muted-foreground">
        <span>Pinned axes</span>
        <span>{mode === "absolute" ? "Full-history hostility" : "Last-turn hostility delta"}</span>
      </div>
    </div>
  );
}
