"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { useDashboard } from "./dashboard-context";

const NAV_ITEMS = [
  {
    href: "/map",
    label: "Map",
    description: "Theater map, focus filters, and entity geography.",
  },
  {
    href: "/monitoring",
    label: "Monitoring",
    description: "Agent posture, reward pressure, and source health.",
  },
  {
    href: "/world",
    label: "World State",
    description: "Coalition graph, traces, and source plan matrix.",
  },
] as const;

export function DashboardShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const {
    runtime,
    session,
    error,
    busy,
    summary,
    createFreshSession,
    toggleLive,
    refreshSources,
    stepSession,
    isInitializingSession,
    isBackendUnavailable,
  } = useDashboard();

  return (
    <div className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Trenches Monitoring Deck</p>
          <h1>Fog-of-War Live Session Dashboard</h1>
          <p className="lede">
            Split into dedicated surfaces so the operator can move between the theater map, model monitoring, and raw
            world state without fighting a single long scroll.
          </p>
        </div>
        <div className="status-card">
          <span>Runtime Booted</span>
          <strong>{runtime?.bootedAt ?? "pending"}</strong>
          <span>Source Validation</span>
          <strong>{runtime ? `${runtime.sourceValidation.duplicateKeys.length} duplicate keys` : "pending"}</strong>
        </div>
      </header>

      <nav className="dashboard-nav" aria-label="Primary">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={isActive ? "page" : undefined}
              className={`dashboard-nav__item${isActive ? " is-active" : ""}`}
            >
              <span className="dashboard-nav__label">{item.label}</span>
              <span className="dashboard-nav__description">{item.description}</span>
            </Link>
          );
        })}
      </nav>

      {children}

      <section className="control-strip">
        <button onClick={createFreshSession} disabled={busy}>
          New Session
        </button>
        <button onClick={() => toggleLive(true)} disabled={busy || !session}>
          Start Live RL Session
        </button>
        <button onClick={() => toggleLive(false)} disabled={busy || !session}>
          Stop Live RL Session
        </button>
        <button onClick={refreshSources} disabled={busy || !session}>
          Refresh Sources
        </button>
        <button onClick={stepSession} disabled={busy || !session}>
          Advance Turn
        </button>
      </section>

      {error ? <section className="banner error">{error}</section> : null}
      {isInitializingSession ? <section className="banner">Bootstrapping live backend session and intelligence overlays.</section> : null}
      {isBackendUnavailable ? (
        <section className="banner">
          Backend is unreachable on <code>http://localhost:8000</code>. The map remains available, but live session data
          and model telemetry will stay offline until the API responds.
        </section>
      ) : null}
      {!session && !isInitializingSession && !isBackendUnavailable ? (
        <section className="banner">No backend session yet. Create one or start the backend.</section>
      ) : null}

      {summary ? (
        <section className="summary-grid">
          {summary.map((item) => (
            <article key={item.label} className="metric">
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </article>
          ))}
        </section>
      ) : null}
    </div>
  );
}
