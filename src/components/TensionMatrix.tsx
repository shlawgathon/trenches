"use client";

import { cn } from "@/src/lib/utils";

export type AgentVisual = {
  flag: string;
  primary: string;
  secondary: string;
};

export type TensionMatrixProps = {
  agents: string[];
  matrix: number[][];
  visuals: Record<string, AgentVisual>;
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
  const clamped = Math.max(0, Math.min(100, value)) / 100;
  const baseBlend = blendHex(rowColor, colColor, 0.5);
  const low = blendHex(baseBlend, "#1e3a8a", 0.55);
  const high = blendHex(baseBlend, "#dc2626", 0.45);
  return blendHex(low, high, clamped);
}

export function TensionMatrix({ agents, matrix, visuals, className }: TensionMatrixProps) {
  if (agents.length === 0) return null;

  return (
    <div className={cn("border border-border/30 bg-card/55 p-3 backdrop-blur-xl", className)}>
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-foreground/80">Inter-Agent Tension Matrix</div>
      <div className="grid" style={{ gridTemplateColumns: `54px repeat(${agents.length}, minmax(0, 1fr))` }}>
        <div />
        {agents.map((agent) => (
          <div key={`col-${agent}`} className="px-1 pb-1 text-center text-[9px] font-mono uppercase text-muted-foreground" title={agent}>
            <span className="mr-1">{visuals[agent]?.flag ?? "🏳️"}</span>
            {agent.slice(0, 3)}
          </div>
        ))}

        {agents.map((rowAgent, rowIdx) => (
          <div key={`row-wrap-${rowAgent}`} className="contents">
            <div key={`row-${rowAgent}`} className="flex items-center pr-1 text-[9px] font-mono uppercase text-muted-foreground">
              <span className="mr-1">{visuals[rowAgent]?.flag ?? "🏳️"}</span>
              {rowAgent.slice(0, 3)}
            </div>
            {agents.map((colAgent, colIdx) => {
              const value = rowIdx === colIdx ? 0 : matrix[rowIdx]?.[colIdx] ?? 0;
              const rowColor = visuals[rowAgent]?.primary ?? "#64748b";
              const colColor = visuals[colAgent]?.secondary ?? "#94a3b8";
              return (
                <div
                  key={`${rowAgent}-${colAgent}`}
                  className="relative m-0.5 h-8 min-w-8 border border-black/20"
                  style={{ backgroundColor: tensionColor(value, rowColor, colColor), opacity: rowIdx === colIdx ? 0.25 : 0.9 }}
                  title={`${visuals[rowAgent]?.flag ?? ""} ${rowAgent} → ${visuals[colAgent]?.flag ?? ""} ${colAgent}: ${value.toFixed(0)}`}
                >
                  <span className="absolute inset-0 flex items-center justify-center text-[9px] font-mono text-white/90">
                    {value.toFixed(0)}
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="mt-2 flex items-center justify-between text-[8px] font-mono uppercase tracking-wider text-muted-foreground">
        <span>Flag cool</span>
        <span>Flag palette blend → hot escalation</span>
        <span>Flag hot</span>
      </div>
    </div>
  );
}
