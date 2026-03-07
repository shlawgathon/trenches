import type { ReactNode } from "react";

import { DashboardProvider } from "../../src/dashboard/dashboard-context";
import { DashboardShell } from "../../src/dashboard/dashboard-shell";

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <DashboardProvider>
      <DashboardShell>{children}</DashboardShell>
    </DashboardProvider>
  );
}
