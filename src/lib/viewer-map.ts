import type { SessionState } from "./types";

import gulfAssets from "../../entities/gulf/assets.json";
import gulfProfile from "../../entities/gulf/profile.json";
import hezbollahAssets from "../../entities/hezbollah/assets.json";
import hezbollahProfile from "../../entities/hezbollah/profile.json";
import iranAssets from "../../entities/iran/assets.json";
import iranProfile from "../../entities/iran/profile.json";
import israelAssets from "../../entities/israel/assets.json";
import israelProfile from "../../entities/israel/profile.json";
import oversightAssets from "../../entities/oversight/assets.json";
import oversightProfile from "../../entities/oversight/profile.json";
import usAssets from "../../entities/us/assets.json";
import usProfile from "../../entities/us/profile.json";

export const MAP_ENTITY_ORDER = ["us", "israel", "iran", "hezbollah", "gulf", "oversight"] as const;

export type MapEntityId = (typeof MAP_ENTITY_ORDER)[number];
export type MapSelection = MapEntityId | "all";
export type ViewerMapLayer =
  | "location"
  | "front"
  | "infrastructure"
  | "strategic_site"
  | "alliance_anchor"
  | "chokepoint"
  | "geospatial_anchor"
  | "coalition_link";

type AssetRecord = Record<string, unknown>;

type EntityAssetPack = {
  locations?: AssetRecord[];
  fronts?: AssetRecord[];
  infrastructure?: AssetRecord[];
  strategic_sites?: AssetRecord[];
  alliance_anchors?: AssetRecord[];
  chokepoints?: AssetRecord[];
  geospatial_anchors?: AssetRecord[];
};

type EntityProfile = {
  display_name?: string;
  home_region?: string;
  strategic_objectives?: string[];
  intelligence_priorities?: string[];
  protected_interests?: string[];
};

type EntityDefinition = {
  id: MapEntityId;
  displayName: string;
  color: string;
  accent: string;
  profile: EntityProfile;
  assets: EntityAssetPack;
};

export type ViewerMapFeature = {
  id: string;
  entityId: MapEntityId;
  entityName: string;
  layer: ViewerMapLayer;
  category: string;
  name: string;
  latitude: number;
  longitude: number;
  description: string;
  sourceKey: string;
};

export type ViewerMapLink = {
  id: string;
  fromAgentId: MapEntityId;
  toAgentId: MapEntityId;
  from: [number, number];
  to: [number, number];
};

export type ViewerMapEntity = {
  id: MapEntityId;
  displayName: string;
  color: string;
  accent: string;
  center: [number, number];
  homeRegion: string;
  headline: string;
  objectives: string[];
  priorities: string[];
  featureCount: number;
  foreignBaseCount: number;
};

export type ViewerMapState = {
  entities: ViewerMapEntity[];
  features: ViewerMapFeature[];
  links: ViewerMapLink[];
  worldSummary: {
    turn: number;
    tension: number;
    marketStress: number;
    oilPressure: number;
    liveMode: boolean;
  };
};

const ENTITY_DEFINITIONS: Record<MapEntityId, EntityDefinition> = {
  us: {
    id: "us",
    displayName: usProfile.display_name,
    color: "#72e6c8",
    accent: "#bff8e9",
    profile: usProfile,
    assets: usAssets,
  },
  israel: {
    id: "israel",
    displayName: israelProfile.display_name,
    color: "#f7c66c",
    accent: "#ffe8ad",
    profile: israelProfile,
    assets: israelAssets,
  },
  iran: {
    id: "iran",
    displayName: iranProfile.display_name,
    color: "#f16f62",
    accent: "#ffcabd",
    profile: iranProfile,
    assets: iranAssets,
  },
  hezbollah: {
    id: "hezbollah",
    displayName: hezbollahProfile.display_name,
    color: "#e87f37",
    accent: "#ffd1a6",
    profile: hezbollahProfile,
    assets: hezbollahAssets,
  },
  gulf: {
    id: "gulf",
    displayName: gulfProfile.display_name,
    color: "#64c6ff",
    accent: "#c8ebff",
    profile: gulfProfile,
    assets: gulfAssets,
  },
  oversight: {
    id: "oversight",
    displayName: oversightProfile.display_name,
    color: "#d68dff",
    accent: "#f1d8ff",
    profile: oversightProfile,
    assets: oversightAssets,
  },
};

const LAYER_DEFINITIONS: Array<{ key: keyof EntityAssetPack; layer: ViewerMapLayer }> = [
  { key: "locations", layer: "location" },
  { key: "fronts", layer: "front" },
  { key: "infrastructure", layer: "infrastructure" },
  { key: "strategic_sites", layer: "strategic_site" },
  { key: "alliance_anchors", layer: "alliance_anchor" },
  { key: "chokepoints", layer: "chokepoint" },
  { key: "geospatial_anchors", layer: "geospatial_anchor" },
];

function toNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function resolveFeatureCoordinates(record: AssetRecord, fallbackLookup: Map<string, [number, number]>): [number, number] | null {
  const lat = toNumber(record.lat);
  const lon = toNumber(record.lon);
  if (lat !== null && lon !== null) {
    return [lon, lat];
  }

  const anchorLat = toNumber(record.anchor_lat);
  const anchorLon = toNumber(record.anchor_lon);
  if (anchorLat !== null && anchorLon !== null) {
    return [anchorLon, anchorLat];
  }

  const linkedLocation = typeof record.location === "string" ? fallbackLookup.get(record.location) : undefined;
  if (linkedLocation) {
    return linkedLocation;
  }

  return null;
}

function buildLocationLookup(assets: EntityAssetPack): Map<string, [number, number]> {
  const lookup = new Map<string, [number, number]>();
  for (const record of assets.locations ?? []) {
    const name = typeof record.name === "string" ? record.name : undefined;
    const coordinates = resolveFeatureCoordinates(record, new Map());
    if (!name || !coordinates) {
      continue;
    }
    lookup.set(name, coordinates);
  }
  return lookup;
}

function buildEntityFeatures(definition: EntityDefinition): ViewerMapFeature[] {
  const locationLookup = buildLocationLookup(definition.assets);
  const features: ViewerMapFeature[] = [];

  for (const layerDefinition of LAYER_DEFINITIONS) {
    const records = definition.assets[layerDefinition.key] ?? [];
    for (const [index, record] of records.entries()) {
      const coordinates = resolveFeatureCoordinates(record, locationLookup);
      if (!coordinates) {
        continue;
      }

      const primaryName =
        typeof record.name === "string"
          ? record.name
          : typeof record.partner === "string" && typeof record.location === "string"
            ? `${record.partner} / ${record.location}`
            : `${definition.displayName} ${layerDefinition.layer} ${index + 1}`;

      const category =
        typeof record.category === "string"
          ? record.category
          : typeof record.type === "string"
            ? record.type
            : layerDefinition.layer;

      const description = [
        typeof record.notes === "string" ? record.notes : null,
        typeof record.goal === "string" ? `Goal: ${record.goal}` : null,
        typeof record.function === "string" ? `Function: ${record.function}` : null,
        Array.isArray(record.operational_focus) && record.operational_focus.length > 0
          ? `Focus: ${record.operational_focus.join(", ")}`
          : null,
      ]
        .filter(Boolean)
        .join(" ");

      features.push({
        id: `${definition.id}-${layerDefinition.layer}-${index}`,
        entityId: definition.id,
        entityName: definition.displayName,
        layer: layerDefinition.layer,
        category,
        name: primaryName,
        longitude: coordinates[0],
        latitude: coordinates[1],
        description,
        sourceKey: layerDefinition.key,
      });
    }
  }

  return features;
}

function averageCoordinates(features: ViewerMapFeature[]): [number, number] {
  if (features.length === 0) {
    return [35.0, 31.0];
  }

  const [lonTotal, latTotal] = features.reduce(
    ([lonSum, latSum], feature) => [lonSum + feature.longitude, latSum + feature.latitude],
    [0, 0],
  );
  return [lonTotal / features.length, latTotal / features.length];
}

function countForeignBases(entityId: MapEntityId, features: ViewerMapFeature[]): number {
  if (entityId !== "us") {
    return 0;
  }

  return features.filter(
    (feature) =>
      feature.layer === "location" &&
      /(air base|naval support activity|camp|diego garcia|jebel ali|patrol box|corridor)/i.test(feature.name),
  ).length;
}

function buildEntities(features: ViewerMapFeature[]): ViewerMapEntity[] {
  return MAP_ENTITY_ORDER.map((entityId) => {
    const definition = ENTITY_DEFINITIONS[entityId];
    const entityFeatures = features.filter((feature) => feature.entityId === entityId);
    const objectives = definition.profile.strategic_objectives ?? [];
    const priorities = definition.profile.intelligence_priorities ?? definition.profile.protected_interests ?? [];

    return {
      id: entityId,
      displayName: definition.displayName,
      color: definition.color,
      accent: definition.accent,
      center: averageCoordinates(entityFeatures),
      homeRegion: definition.profile.home_region ?? "regional",
      headline: objectives[0] ?? "Monitor and adapt to the theater picture.",
      objectives,
      priorities: priorities.slice(0, 4),
      featureCount: entityFeatures.length,
      foreignBaseCount: countForeignBases(entityId, entityFeatures),
    };
  });
}

function buildLinks(session: SessionState, entities: ViewerMapEntity[]): ViewerMapLink[] {
  const centerLookup = new Map<MapEntityId, [number, number]>(
    entities.map((entity) => [entity.id, entity.center]),
  );
  const seen = new Set<string>();
  const links: ViewerMapLink[] = [];

  for (const source of MAP_ENTITY_ORDER) {
    const targets = session.world.coalition_graph[source] ?? [];
    for (const target of targets) {
      if (!MAP_ENTITY_ORDER.includes(target as MapEntityId)) {
        continue;
      }

      const targetId = target as MapEntityId;
      const key = [source, targetId].sort().join(":");
      if (seen.has(key)) {
        continue;
      }

      const from = centerLookup.get(source);
      const to = centerLookup.get(targetId);
      if (!from || !to) {
        continue;
      }

      seen.add(key);
      links.push({
        id: `coalition-${key}`,
        fromAgentId: source,
        toAgentId: targetId,
        from,
        to,
      });
    }
  }

  return links;
}

export function buildViewerMapState(session: SessionState): ViewerMapState {
  const features = MAP_ENTITY_ORDER.flatMap((entityId) => buildEntityFeatures(ENTITY_DEFINITIONS[entityId]));
  const entities = buildEntities(features);

  return {
    entities,
    features,
    links: buildLinks(session, entities),
    worldSummary: {
      turn: session.world.turn,
      tension: session.world.tension_level,
      marketStress: session.world.market_stress,
      oilPressure: session.world.oil_pressure,
      liveMode: session.live.enabled,
    },
  };
}
