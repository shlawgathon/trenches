import { HttpClient } from "./http";
import type {
  CreateSessionRequest,
  LiveControlRequest,
  ResetSessionRequest,
  SessionState,
  StepSessionRequest,
  StepSessionResponse,
} from "./types";

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

  async setLiveMode(sessionId: string, request: LiveControlRequest): Promise<SessionState> {
    return this.http.post<SessionState>(`/sessions/${sessionId}/live`, request);
  }

  async stepSession(sessionId: string, request: StepSessionRequest): Promise<StepSessionResponse> {
    return this.http.post<StepSessionResponse>(`/sessions/${sessionId}/step`, request);
  }
}
