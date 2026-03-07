import { useEffect, useId, useMemo, useRef, useState, type CSSProperties } from "react";
import mapboxgl, {
  type GeoJSONSource,
  type LngLatBoundsLike,
  type MapboxGeoJSONFeature,
  type MapMouseEvent,
} from "mapbox-gl";

import { CommandGlobe } from "./CommandGlobe";
import type { MapSelection, ViewerMapEntity, ViewerMapFeature, ViewerMapLink, ViewerMapState } from "../lib/viewer-map";

type CommandMapProps = {
  entities: ViewerMapEntity[];
  features: ViewerMapFeature[];
  links?: ViewerMapLink[];
  selectedEntity: MapSelection;
  onSelectEntity: (entityId: MapSelection) => void;
  worldSummary: ViewerMapState["worldSummary"] & {
    activeEventCount?: number;
    lastUpdatedLabel?: string;
  };
};

type GeoJsonFeatureProperties = {
  id: string;
  entityId: string;
  entityLabel: string;
  title: string;
  subtitle: string;
  category: string;
  importance: string;
  color: string;
  accent: string;
  selected: number;
  dimmed: number;
  interactive: number;
};

type GeoJsonFeatureCollection = GeoJSON.FeatureCollection<GeoJSON.Geometry, GeoJsonFeatureProperties>;
type MapFeatureMouseEvent = MapMouseEvent & { features?: MapboxGeoJSONFeature[] };

const DEFAULT_CENTER: [number, number] = [41.8, 27.8];
const DEFAULT_ZOOM = 2.35;
const FALLBACK_ENTITY: MapSelection = "all";
const MAP_STYLE = "mapbox://styles/mapbox/dark-v11";
const INTEL_HIDDEN_LAYERS = [
  "road-label",
  "settlement-label",
  "settlement-subdivision-label",
  "airport-label",
  "poi-label",
  "transit-label",
  "natural-label",
];
const LAYER_LABELS: Record<string, string> = {
  alliance_anchor: "Alliance Anchor",
  chokepoint: "Chokepoint",
  coalition_link: "Coalition Link",
  front: "Front",
  geospatial_anchor: "Geospatial Anchor",
  infrastructure: "Infrastructure",
  location: "Location",
  strategic_site: "Strategic Site",
};

export function CommandMap({
  entities,
  features,
  links = [],
  selectedEntity,
  onSelectEntity,
  worldSummary,
}: CommandMapProps) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const popupRef = useRef<mapboxgl.Popup | null>(null);
  const [interactiveReady, setInteractiveReady] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  const mapId = useId().replace(/:/g, "");
  const sourceId = `trenches-command-map-source-${mapId}`;
  const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;
  const hasToken = typeof token === "string" && token.trim().length > 0;

  const entityLookup = useMemo(
    () => new Map(entities.map((entity) => [entity.id, entity])),
    [entities],
  );

  const geoJson = useMemo<GeoJsonFeatureCollection>(() => {
    const pointFeatures: GeoJSON.Feature<GeoJSON.Point, GeoJsonFeatureProperties>[] = features.map((feature) => {
      const entity = entityLookup.get(feature.entityId);
      const isSelected = selectedEntity === FALLBACK_ENTITY || selectedEntity === feature.entityId;
      return {
        type: "Feature",
        id: feature.id,
        properties: {
          id: feature.id,
          entityId: feature.entityId,
          entityLabel: feature.entityName,
          title: feature.name,
          subtitle: feature.description,
          category: feature.layer,
          importance: feature.layer === "chokepoint" ? "critical" : feature.layer === "front" ? "high" : "medium",
          color: entity?.color ?? "#90d1ff",
          accent: entity?.accent ?? "#dff3fb",
          selected: isSelected ? 1 : 0,
          dimmed: isSelected ? 0 : 1,
          interactive: 1,
        },
        geometry: {
          type: "Point",
          coordinates: [feature.longitude, feature.latitude],
        },
      };
    });

    const linkFeatures: GeoJSON.Feature<GeoJSON.LineString, GeoJsonFeatureProperties>[] = links.map((link) => {
      const entity = entityLookup.get(link.fromAgentId);
      const isSelected =
        selectedEntity === FALLBACK_ENTITY || selectedEntity === link.fromAgentId || selectedEntity === link.toAgentId;
      const fromLabel = entityLookup.get(link.fromAgentId)?.displayName ?? link.fromAgentId;
      const toLabel = entityLookup.get(link.toAgentId)?.displayName ?? link.toAgentId;
      return {
        type: "Feature",
        id: link.id,
        properties: {
          id: link.id,
          entityId: link.fromAgentId,
          entityLabel: `${fromLabel} ↔ ${toLabel}`,
          title: "Coalition Link",
          subtitle: `${fromLabel} coordinated with ${toLabel}.`,
          category: "coalition_link",
          importance: "medium",
          color: entity?.color ?? "#90d1ff",
          accent: entity?.accent ?? "#dff3fb",
          selected: isSelected ? 1 : 0,
          dimmed: isSelected ? 0 : 1,
          interactive: 0,
        },
        geometry: {
          type: "LineString",
          coordinates: [link.from, link.to],
        },
      };
    });

    return {
      type: "FeatureCollection",
      features: [...linkFeatures, ...pointFeatures],
    };
  }, [entityLookup, features, links, selectedEntity]);

  const focusFeatures = useMemo(() => {
    if (selectedEntity === FALLBACK_ENTITY) {
      return features;
    }
    return features.filter((feature) => feature.entityId === selectedEntity);
  }, [features, selectedEntity]);

  const focusLinks = useMemo(() => {
    if (selectedEntity === FALLBACK_ENTITY) {
      return links;
    }
    return links.filter((link) => link.fromAgentId === selectedEntity || link.toAgentId === selectedEntity);
  }, [links, selectedEntity]);

  const categorySummary = useMemo(() => {
    const counts = new Map<string, number>();
    for (const feature of focusFeatures) {
      counts.set(feature.layer, (counts.get(feature.layer) ?? 0) + 1);
    }
    return [...counts.entries()]
      .sort((left, right) => right[1] - left[1])
      .slice(0, 5)
      .map(([layer, count]) => ({
        layer,
        label: LAYER_LABELS[layer] ?? startCase(layer),
        count,
      }));
  }, [focusFeatures]);

  const selectedEntityMeta = useMemo(() => {
    if (selectedEntity === FALLBACK_ENTITY) {
      return {
        label: "All Entities",
        color: "linear-gradient(135deg, #87f8c1 0%, #6bb8ff 100%)",
        headline: "Viewer-wide synthesis of every actor footprint, infrastructure chain, coalition line, and chokepoint.",
        priorities: ["Cross-entity route awareness", "Coalition edges", "Escalation geography", "Foreign basing"],
        featureCount: features.length,
        foreignBaseCount: 0,
      };
    }
    const entity =
      selectedEntity === FALLBACK_ENTITY ? undefined : entityLookup.get(selectedEntity as ViewerMapEntity["id"]);
    return {
      label: entity?.displayName ?? selectedEntity,
      color: entity?.color ?? "#90d1ff",
      headline: entity?.headline ?? "Monitor the mapped footprint and strategic posture.",
      priorities: entity?.priorities ?? [],
      featureCount: entity?.featureCount ?? focusFeatures.length,
      foreignBaseCount: entity?.foreignBaseCount ?? 0,
    };
  }, [entityLookup, features.length, focusFeatures.length, selectedEntity]);

  useEffect(() => {
    if (!hasToken || !mapContainerRef.current || mapRef.current) {
      return;
    }

    setInteractiveReady(false);
    setMapError(null);
    mapboxgl.accessToken = token;
    let map: mapboxgl.Map;
    try {
      map = new mapboxgl.Map({
        container: mapContainerRef.current,
        style: MAP_STYLE,
        center: DEFAULT_CENTER,
        zoom: DEFAULT_ZOOM,
        attributionControl: false,
        projection: "globe",
      });
    } catch (error) {
      setMapError(error instanceof Error ? error.message : "Failed to initialize WebGL.");
      return;
    }

    mapRef.current = map;
    popupRef.current = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false,
      offset: 18,
      className: "command-map-popup",
    });

    map.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), "bottom-right");

    map.on("style.load", () => {
      applyIntelligenceTheme(map);
      map.setFog({
        color: "rgba(1, 5, 8, 0.96)",
        "high-color": "rgba(9, 27, 39, 0.34)",
        "space-color": "rgba(2, 5, 8, 1)",
        "star-intensity": 0.08,
        range: [-1, 2],
      });
    });

    map.on("load", () => {
      setInteractiveReady(true);
      if (!map.getSource(sourceId)) {
        map.addSource(sourceId, {
          type: "geojson",
          data: geoJson,
        });
      }

      map.addLayer({
        id: `${sourceId}-lines`,
        type: "line",
        source: sourceId,
        filter: ["==", ["geometry-type"], "LineString"],
        paint: {
          "line-color": ["get", "color"],
          "line-width": ["case", ["==", ["get", "selected"], 1], 3.2, 1.8],
          "line-opacity": ["case", ["==", ["get", "selected"], 1], 0.9, 0.24],
          "line-blur": ["case", ["==", ["get", "selected"], 1], 0.28, 0.85],
        },
      });

      map.addLayer({
        id: `${sourceId}-point-halo`,
        type: "circle",
        source: sourceId,
        filter: ["==", ["geometry-type"], "Point"],
        paint: {
          "circle-radius": ["case", ["==", ["get", "selected"], 1], 15, 9],
          "circle-color": ["get", "color"],
          "circle-opacity": ["case", ["==", ["get", "selected"], 1], 0.2, 0.05],
          "circle-blur": 0.58,
        },
      });

      map.addLayer({
        id: `${sourceId}-points`,
        type: "circle",
        source: sourceId,
        filter: ["==", ["geometry-type"], "Point"],
        paint: {
          "circle-radius": [
            "case",
            ["==", ["get", "importance"], "critical"],
            6.8,
            ["==", ["get", "importance"], "high"],
            5.9,
            4.9,
          ],
          "circle-color": ["get", "color"],
          "circle-stroke-color": ["case", ["==", ["get", "selected"], 1], "#f8fdff", "rgba(255,255,255,0.42)"],
          "circle-stroke-width": ["case", ["==", ["get", "selected"], 1], 1.45, 0.8],
          "circle-opacity": ["case", ["==", ["get", "selected"], 1], 0.98, 0.34],
        },
      });

      map.addLayer({
        id: `${sourceId}-labels`,
        type: "symbol",
        source: sourceId,
        filter: ["all", ["==", ["geometry-type"], "Point"], ["==", ["get", "selected"], 1]],
        layout: {
          "text-field": ["get", "title"],
          "text-size": 11,
          "text-offset": [0, 1.28],
          "text-anchor": "top",
          "text-font": ["DIN Pro Medium", "Arial Unicode MS Regular"],
        },
        paint: {
          "text-color": "rgba(233, 244, 248, 0.92)",
          "text-halo-color": "rgba(4, 10, 15, 0.95)",
          "text-halo-width": 1.2,
          "text-opacity": 0.9,
        },
      });

      bindInteraction(map, popupRef.current, onSelectEntity);
      fitMapToSelection(map, focusFeatures, focusLinks);
    });

    map.on("error", (event) => {
      const nextError = event.error;
      setMapError(nextError instanceof Error ? nextError.message : "Map rendering error.");
    });

    return () => {
      popupRef.current?.remove();
      popupRef.current = null;
      setInteractiveReady(false);
      setMapError(null);
      map.remove();
      mapRef.current = null;
    };
  }, [focusFeatures, focusLinks, geoJson, hasToken, onSelectEntity, sourceId, token]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }
    const source = map.getSource(sourceId) as GeoJSONSource | undefined;
    if (source) {
      source.setData(geoJson);
    }
    fitMapToSelection(map, focusFeatures, focusLinks);
  }, [focusFeatures, focusLinks, geoJson, sourceId]);

  return (
    <section className="command-map">
      <div className="command-map__header">
        <div>
          <p className="command-map__eyebrow">Viewer Layer</p>
          <h2>World Intelligence Layer</h2>
          <p className="command-map__lede">
            This is a spectator command surface. Agents do not receive this spatial synthesis in their observation
            payload.
          </p>
        </div>
        <div className="command-map__metrics">
          <MetricCard label="Turn" value={String(worldSummary.turn)} tone="neutral" />
          <MetricCard label="Tension" value={worldSummary.tension.toFixed(1)} tone={scoreTone(worldSummary.tension)} />
          <MetricCard
            label="Market Stress"
            value={worldSummary.marketStress.toFixed(1)}
            tone={scoreTone(worldSummary.marketStress)}
          />
          <MetricCard label="Oil" value={worldSummary.oilPressure.toFixed(1)} tone={scoreTone(worldSummary.oilPressure)} />
          <MetricCard label="Live" value={worldSummary.liveMode ? "Armed" : "Static"} tone="neutral" />
          <MetricCard label="Events" value={String(worldSummary.activeEventCount ?? 0)} tone="neutral" />
        </div>
      </div>

      <div className="command-map__viewport">
        <div className="command-map__surface">
          {hasToken && !mapError ? (
            <div ref={mapContainerRef} className="command-map__canvas" aria-label="Operational asset map" />
          ) : !hasToken ? (
            <div className="command-map__fallback" role="status" aria-live="polite">
              <div className="command-map__fallback-badge">Mapbox token missing</div>
              <h3>Spatial viewer is ready, but the tile layer is offline.</h3>
              <p>
                Set <code>NEXT_PUBLIC_MAPBOX_TOKEN</code> to render the theater map. Entity footprints, coalition links, and
                viewer controls remain available.
              </p>
              <div className="command-map__fallback-grid">
                <div>
                  <span>Tracked footprints</span>
                  <strong>{features.length}</strong>
                </div>
                <div>
                  <span>Coalition links</span>
                  <strong>{links.length}</strong>
                </div>
              </div>
            </div>
          ) : (
            <CommandGlobe
              entities={entities}
              features={focusFeatures}
              links={focusLinks}
              selectedEntity={selectedEntity}
              onSelectEntity={onSelectEntity}
            />
          )}
          {hasToken && mapError ? (
            <div className="command-map__render-badge">WebGL unavailable, globe fallback active</div>
          ) : hasToken && !interactiveReady ? (
            <div className="command-map__render-badge">Initializing interactive globe</div>
          ) : null}

          <div className="command-map__overlay command-map__overlay--top">
            <div className="command-map__viewer-chip">Operator intelligence wall</div>
            {worldSummary.lastUpdatedLabel ? <span className="command-map__timestamp">{worldSummary.lastUpdatedLabel}</span> : null}
          </div>

          <div className="command-map__overlay command-map__overlay--bottom">
            <div className="command-map__legend-line">
              <span className="command-map__legend-title">Legend</span>
              <span>Signals mark assets, fronts, chokepoints, and foreign basing. Lines trace active coalition edges.</span>
            </div>
            <div className="command-map__legend-swatches">
              {entities.map((entity) => (
                <button
                  key={entity.id}
                  type="button"
                  className={`command-map__swatch${selectedEntity === entity.id ? " is-active" : ""}`}
                  onClick={() => onSelectEntity(entity.id)}
                  style={cssVarStyle("--command-swatch", entity.color)}
                >
                  <span className="command-map__swatch-color" />
                  <span>{entity.displayName}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        <aside className="command-map__panel">
          <div className="command-map__notice">
            <span className="command-map__notice-label">Fog-of-war boundary</span>
            <p>
              The viewer sees the merged theater footprint. Models only receive text briefs, scoped source bundles, and
              role-limited actions.
            </p>
          </div>

          <div className="command-map__selector">
            <label htmlFor={`${mapId}-entity-selector`}>Focus Entity</label>
            <div className="command-map__selector-grid">
              <button
                type="button"
                className={`command-map__entity-button${selectedEntity === FALLBACK_ENTITY ? " is-active" : ""}`}
                onClick={() => onSelectEntity(FALLBACK_ENTITY)}
              >
                All Entities
              </button>
              {entities.map((entity) => (
                <button
                  key={entity.id}
                  type="button"
                  className={`command-map__entity-button${selectedEntity === entity.id ? " is-active" : ""}`}
                  onClick={() => onSelectEntity(entity.id)}
                  style={cssVarStyle("--command-entity-color", entity.color)}
                >
                  {entity.displayName}
                </button>
              ))}
            </div>
          </div>

          <div className="command-map__focus-card">
            <div className="command-map__focus-bar" style={{ background: selectedEntityMeta.color }} />
            <p className="command-map__focus-label">Active Focus</p>
            <h3>{selectedEntityMeta.label}</h3>
            <p>{selectedEntityMeta.headline}</p>
            {selectedEntityMeta.priorities.length > 0 ? (
              <ul className="command-map__priority-list">
                {selectedEntityMeta.priorities.slice(0, 4).map((priority) => (
                  <li key={priority}>{priority}</li>
                ))}
              </ul>
            ) : null}
            <div className="command-map__focus-stats">
              <div>
                <span>Footprints</span>
                <strong>{selectedEntityMeta.featureCount}</strong>
              </div>
              <div>
                <span>Foreign Bases</span>
                <strong>{selectedEntityMeta.foreignBaseCount}</strong>
              </div>
            </div>
          </div>

          <div className="command-map__category-stack">
            <div className="command-map__section-head">
              <h3>Top Layers</h3>
              <span>{selectedEntity === FALLBACK_ENTITY ? "Entire theater" : "Focused entity"}</span>
            </div>
            {categorySummary.length > 0 ? (
              categorySummary.map((item) => (
                <div key={item.layer} className="command-map__category-row">
                  <span>{item.label}</span>
                  <strong>{item.count}</strong>
                </div>
              ))
            ) : (
              <p className="command-map__empty">No mapped footprints yet.</p>
            )}
          </div>

          <div className="command-map__entity-list">
            <div className="command-map__section-head">
              <h3>Entities</h3>
              <span>Mapped routing</span>
            </div>
            {entities.map((entity) => (
              <button
                key={entity.id}
                type="button"
                className={`command-map__entity-row${selectedEntity === entity.id ? " is-active" : ""}`}
                onClick={() => onSelectEntity(entity.id)}
                style={cssVarStyle("--command-entity-color", entity.color)}
              >
                <span className="command-map__entity-color" />
                <span className="command-map__entity-copy">
                  <strong>{entity.displayName}</strong>
                  <small>{entity.headline}</small>
                </span>
                <span className="command-map__entity-count">{entity.featureCount}</span>
              </button>
            ))}
          </div>
        </aside>
      </div>
    </section>
  );
}

function MetricCard({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "neutral" | "warn" | "hot";
}) {
  return (
    <div className={`command-map__metric command-map__metric--${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function scoreTone(value: number): "neutral" | "warn" | "hot" {
  if (value >= 75) {
    return "hot";
  }
  if (value >= 50) {
    return "warn";
  }
  return "neutral";
}

function applyIntelligenceTheme(map: mapboxgl.Map) {
  for (const layerId of INTEL_HIDDEN_LAYERS) {
    if (map.getLayer(layerId)) {
      map.setLayoutProperty(layerId, "visibility", "none");
    }
  }

  const paintUpdates: Array<{
    id: string;
    prop: string;
    value: string | number;
  }> = [
    { id: "background", prop: "background-color", value: "#02070a" },
    { id: "land", prop: "fill-color", value: "#071117" },
    { id: "water", prop: "fill-color", value: "#040b10" },
    { id: "waterway", prop: "line-color", value: "#0d2732" },
    { id: "admin-0-boundary", prop: "line-color", value: "rgba(90, 150, 170, 0.18)" },
    { id: "admin-1-boundary", prop: "line-color", value: "rgba(90, 150, 170, 0.1)" },
    { id: "road-primary", prop: "line-color", value: "rgba(65, 96, 108, 0.16)" },
    { id: "road-secondary-tertiary", prop: "line-color", value: "rgba(55, 78, 88, 0.1)" },
    { id: "airport", prop: "circle-color", value: "rgba(120, 170, 188, 0.18)" },
    { id: "country-label", prop: "text-color", value: "rgba(174, 204, 214, 0.38)" },
    { id: "marine-label", prop: "text-color", value: "rgba(79, 121, 138, 0.32)" },
  ];

  for (const update of paintUpdates) {
    if (!map.getLayer(update.id)) {
      continue;
    }
    try {
      map.setPaintProperty(update.id, update.prop as never, update.value as never);
    } catch {
      continue;
    }
  }
}

function bindInteraction(
  map: mapboxgl.Map,
  popup: mapboxgl.Popup | null,
  onSelectEntity: (entityId: MapSelection) => void,
) {
  const interactiveLayers = map
    .getStyle()
    .layers?.filter((layer) => layer.id.endsWith("-points") || layer.id.endsWith("-lines"))
    .map((layer) => layer.id);

  if (!interactiveLayers || interactiveLayers.length === 0) {
    return;
  }

  for (const layerId of interactiveLayers) {
    map.on("mouseenter", layerId, () => {
      map.getCanvas().style.cursor = "pointer";
    });

    map.on("mouseleave", layerId, () => {
      map.getCanvas().style.cursor = "";
      popup?.remove();
    });

    map.on("mousemove", layerId, (event) => {
      renderPopup(map, popup, event);
    });

    map.on("click", layerId, (event) => {
      const feature = event.features?.[0];
      if (!feature || !feature.properties) {
        return;
      }
      const properties = feature.properties as GeoJsonFeatureProperties;
      if (properties.interactive === 1) {
        onSelectEntity(properties.entityId as MapSelection);
      }
    });
  }
}

function renderPopup(map: mapboxgl.Map, popup: mapboxgl.Popup | null, event: MapFeatureMouseEvent) {
  if (!popup) {
    return;
  }
  const feature = event.features?.[0];
  if (!feature || !feature.properties) {
    popup.remove();
    return;
  }

  const properties = feature.properties as GeoJsonFeatureProperties;
  popup
    .setLngLat(resolvePopupCoordinates(feature.geometry))
    .setHTML(buildPopupMarkup(properties))
    .addTo(map);
}

function buildPopupMarkup(properties: GeoJsonFeatureProperties) {
  return [
    `<div class="command-map-popup__card">`,
    `<span class="command-map-popup__category">${escapeHtml(LAYER_LABELS[properties.category] ?? startCase(properties.category))}</span>`,
    `<strong>${escapeHtml(properties.title)}</strong>`,
    `<p>${escapeHtml(properties.entityLabel)}</p>`,
    properties.subtitle ? `<p class="command-map-popup__subtitle">${escapeHtml(properties.subtitle)}</p>` : "",
    `</div>`,
  ].join("");
}

function resolvePopupCoordinates(geometry: GeoJSON.Geometry): [number, number] {
  if (geometry.type === "Point") {
    return geometry.coordinates as [number, number];
  }
  if (geometry.type === "LineString") {
    return (geometry.coordinates[0] as [number, number]) ?? DEFAULT_CENTER;
  }
  return DEFAULT_CENTER;
}

function fitMapToSelection(map: mapboxgl.Map, features: ViewerMapFeature[], links: ViewerMapLink[]) {
  const coordinates: Array<[number, number]> = [
    ...features.map((feature) => [feature.longitude, feature.latitude] as [number, number]),
    ...links.flatMap((link) => [link.from, link.to]),
  ];

  if (coordinates.length === 0) {
    map.easeTo({ center: DEFAULT_CENTER, zoom: DEFAULT_ZOOM, duration: 650 });
    return;
  }
  if (coordinates.length === 1) {
    map.easeTo({ center: coordinates[0], zoom: 4.2, duration: 650 });
    return;
  }

  const bounds = new mapboxgl.LngLatBounds(coordinates[0], coordinates[0]);
  for (const coordinate of coordinates.slice(1)) {
    bounds.extend(coordinate);
  }
  map.fitBounds(bounds as LngLatBoundsLike, {
    padding: { top: 80, right: 420, bottom: 80, left: 80 },
    duration: 850,
    maxZoom: 5.3,
  });
}

function startCase(value: string) {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function escapeHtml(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function cssVarStyle(name: string, value: string): CSSProperties {
  return {
    [name]: value,
  } as CSSProperties;
}

export default CommandMap;
