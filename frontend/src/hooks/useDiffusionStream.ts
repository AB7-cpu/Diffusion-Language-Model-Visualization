import { useCallback, useEffect, useRef, useState } from "react";

export type TokenState = {
  id: number;
  text: string;
  masked: boolean;
};

type StepEvent = { type: "step"; step: number; total_steps: number; tokens: TokenState[] };
type CompleteEvent = { type: "complete" };
type ErrorEvent = { type: "error"; message: string };
type ServerEvent = StepEvent | CompleteEvent | ErrorEvent;

export type GenerationParams = {
  prompt: string;
  steps: number;
  gen_length: number;
  block_length: number;
  temperature: number;
  remasking: string;
};

export type StreamStatus = "idle" | "connecting" | "streaming" | "complete" | "error";

/**
 * Owns one WebSocket connection to /ws/generate at a time. Calling start()
 * closes any prior connection, opens a fresh one, sends the generation
 * params as soon as it's open, and updates `tokens` in place as "step"
 * events arrive — matching the server's one-generation-per-connection
 * design (see backend/server.py).
 */
export function useDiffusionStream(wsUrl: string) {
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [tokens, setTokens] = useState<TokenState[]>([]);
  const [step, setStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const start = useCallback(
    (params: GenerationParams) => {
      wsRef.current?.close();

      setStatus("connecting");
      setTokens([]);
      setStep(0);
      setTotalSteps(0);
      setError(null);

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("streaming");
        ws.send(JSON.stringify(params));
      };

      ws.onmessage = (event) => {
        const data: ServerEvent = JSON.parse(event.data);
        if (data.type === "step") {
          setStep(data.step);
          setTotalSteps(data.total_steps);
          setTokens(data.tokens);
        } else if (data.type === "complete") {
          setStatus("complete");
        } else if (data.type === "error") {
          setStatus("error");
          setError(data.message);
        }
      };

      ws.onerror = () => {
        setStatus("error");
        setError("Could not reach the generation server. Is server.py running?");
      };

      ws.onclose = () => {
        // A socket that closes while we're still "streaming" (rather than
        // having already reached "complete"/"error") means it dropped
        // unexpectedly — don't leave the UI stuck showing a spinner.
        setStatus((current) => (current === "streaming" ? "error" : current));
      };
    },
    [wsUrl]
  );

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return { status, tokens, step, totalSteps, error, start };
}
