"use client";

import { CommandMap } from "../components/CommandMap";

import { useDashboard } from "./dashboard-context";

export function MapPage() {
  const {
    commandMapEntities,
    commandMapFeatures,
    commandMapLinks,
    commandMapWorldSummary,
    selectedMapEntity,
    setSelectedMapEntity,
  } = useDashboard();

  return (
    <CommandMap
      entities={commandMapEntities}
      features={commandMapFeatures}
      links={commandMapLinks}
      selectedEntity={selectedMapEntity}
      onSelectEntity={setSelectedMapEntity}
      worldSummary={commandMapWorldSummary}
    />
  );
}
