"use client";

import { geoGraticule10, geoOrthographic, geoPath } from "d3-geo";
import { feature } from "topojson-client";
import worldAtlas from "world-atlas/countries-110m.json";
import { useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

import type { MapSelection, ViewerMapEntity, ViewerMapFeature, ViewerMapLink } from "../lib/viewer-map";

type CommandGlobeProps = {
  entities: ViewerMapEntity[];
  features: ViewerMapFeature[];
  links?: ViewerMapLink[];
  selectedEntity: MapSelection;
  onSelectEntity: (entityId: MapSelection) => void;
};

const SVG_WIDTH = 1280;
const SVG_HEIGHT = 720;

const countries = feature(
  worldAtlas as unknown as Parameters<typeof feature>[0],
  (worldAtlas as { objects: { countries: object } }).objects.countries as Parameters<typeof feature>[1],
) as unknown as GeoJSON.FeatureCollection;

export function CommandGlobe({
  entities,
  features,
  links = [],
  selectedEntity,
  onSelectEntity,
}: CommandGlobeProps) {
  const [rotation, setRotation] = useState<[number, number, number]>([-20, -18, 0]);
  const dragRef = useRef<{ x: number; y: number; rotation: [number, number, number] } | null>(null);

  const entityLookup = useMemo(() => new Map(entities.map((entity) => [entity.id, entity])), [entities]);
  const projection = useMemo(() => {
    return geoOrthographic()
      .translate([SVG_WIDTH / 2, SVG_HEIGHT / 2])
      .scale(300)
      .clipAngle(90)
      .rotate(rotation);
  }, [rotation]);

  const path = useMemo(() => geoPath(projection), [projection]);
  const spherePath = path({ type: "Sphere" });
  const graticulePath = path(geoGraticule10());

  const projectedPoints = useMemo(() => {
    return features
      .map((item) => {
        const point = projection([item.longitude, item.latitude]);
        const entity = entityLookup.get(item.entityId);
        const isSelected = selectedEntity === "all" || selectedEntity === item.entityId;
        return {
          ...item,
          point,
          color: entity?.color ?? "#90d1ff",
          accent: entity?.accent ?? "#dff3fb",
          isSelected,
        };
      })
      .filter((item) => item.point);
  }, [entityLookup, features, projection, selectedEntity]);

  const projectedLinks = useMemo(() => {
    return links
      .map((link) => {
        const linePath = path({
          type: "Feature",
          geometry: {
            type: "LineString",
            coordinates: [link.from, link.to],
          },
          properties: {},
        });
        const isSelected =
          selectedEntity === "all" || selectedEntity === link.fromAgentId || selectedEntity === link.toAgentId;
        return {
          ...link,
          linePath,
          color: entityLookup.get(link.fromAgentId)?.color ?? "#90d1ff",
          isSelected,
        };
      })
      .filter((link) => Boolean(link.linePath));
  }, [entityLookup, links, path, selectedEntity]);

  function handlePointerDown(event: ReactPointerEvent<SVGSVGElement>) {
    dragRef.current = {
      x: event.clientX,
      y: event.clientY,
      rotation,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event: ReactPointerEvent<SVGSVGElement>) {
    if (!dragRef.current) {
      return;
    }
    const deltaX = event.clientX - dragRef.current.x;
    const deltaY = event.clientY - dragRef.current.y;
    setRotation([
      dragRef.current.rotation[0] + deltaX * 0.25,
      Math.max(-55, Math.min(55, dragRef.current.rotation[1] - deltaY * 0.2)),
      0,
    ]);
  }

  function handlePointerUp(event: ReactPointerEvent<SVGSVGElement>) {
    dragRef.current = null;
    event.currentTarget.releasePointerCapture(event.pointerId);
  }

  return (
    <div className="command-globe" aria-label="Operational globe fallback">
      <svg
        viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
        className="command-globe__svg"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
      >
        <defs>
          <radialGradient id="command-globe-ocean" cx="50%" cy="45%" r="70%">
            <stop offset="0%" stopColor="#143748" />
            <stop offset="65%" stopColor="#091926" />
            <stop offset="100%" stopColor="#03090d" />
          </radialGradient>
          <filter id="command-globe-glow">
            <feGaussianBlur stdDeviation="8" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {spherePath ? <path d={spherePath} className="command-globe__sphere" /> : null}
        {graticulePath ? <path d={graticulePath} className="command-globe__graticule" /> : null}

        {countries.features.map((country, index) => {
          const countryPath = path(country);
          if (!countryPath) {
            return null;
          }
          return <path key={index} d={countryPath} className="command-globe__land" />;
        })}

        {projectedLinks.map((link) => (
          <path
            key={link.id}
            d={link.linePath ?? ""}
            className={`command-globe__link${link.isSelected ? " is-selected" : ""}`}
            style={{ ["--command-globe-link" as never]: link.color }}
          />
        ))}

        {projectedPoints.map((item) => {
          const [x, y] = item.point as [number, number];
          return (
            <g
              key={item.id}
              className={`command-globe__point${item.isSelected ? " is-selected" : ""}`}
              transform={`translate(${x} ${y})`}
              onClick={() => onSelectEntity(item.entityId)}
            >
              <circle className="command-globe__point-halo" r={item.isSelected ? 12 : 8} style={{ fill: item.color }} />
              <circle className="command-globe__point-core" r={item.isSelected ? 4.8 : 3.2} style={{ fill: item.accent }} />
            </g>
          );
        })}
      </svg>

      <div className="command-globe__hud">
        <span>Interactive globe fallback</span>
        <span>Drag to rotate</span>
      </div>
    </div>
  );
}
