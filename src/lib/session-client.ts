import { HttpClient } from "./http";
import type {
  CreateSessionRequest,
  ExternalSignal,
  LiveControlRequest,
  ResetSessionRequest,
  SourceMonitorReport,
  SessionState,
  StepSessionRequest,
  StepSessionResponse,
} from "./types";

export type IngestNewsRequest = {
  signals: ExternalSignal[];
  agent_ids?: string[];
};

export type IngestNewsResponse = {
  session: SessionState;
  oversight: { triggered: boolean; risk_score: number; reason: string };
  reaction: unknown | null;
  done: boolean;
};

export class SessionClient {
  constructor(private readonly http: HttpClient) {}

  async health(): Promise<{ status: string }> {
    return this.http.get<{ status: string }>("/healthz");
  }

  async createSession(request: CreateSessionRequest = {}): Promise<SessionState> {
    return this.http.post<SessionState>("/sessions", request);
  }

  async getSession(sessionId: string): Promise<SessionState> {
    return this.http.get<SessionState>(`/sessions/${sessionId}`);
  }

  async resetSession(sessionId: string, request: ResetSessionRequest = {}): Promise<SessionState> {
    return this.http.post<SessionState>(`/sessions/${sessionId}/reset`, request);
  }

  async refreshSources(sessionId: string): Promise<SessionState> {
    return this.http.post<SessionState>(`/sessions/${sessionId}/sources/refresh`, {});
  }

  async getSourceMonitor(sessionId: string): Promise<SourceMonitorReport> {
    return this.http.get<SourceMonitorReport>(`/sessions/${sessionId}/sources/monitor`);
  }

  async setLiveMode(sessionId: string, request: LiveControlRequest): Promise<SessionState> {
    return this.http.post<SessionState>(`/sessions/${sessionId}/live`, request);
  }

  async stepSession(sessionId: string, request: StepSessionRequest): Promise<StepSessionResponse> {
    return this.http.post<StepSessionResponse>(`/sessions/${sessionId}/step`, request);
  }

  async ingestNews(sessionId: string, request: IngestNewsRequest): Promise<IngestNewsResponse> {
    return this.http.post<IngestNewsResponse>(`/sessions/${sessionId}/news`, request);
  }
}
