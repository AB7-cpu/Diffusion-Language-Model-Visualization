import { useState } from "react";
import type { FormEvent } from "react";
import { useDiffusionStream } from "./hooks/useDiffusionStream";
import TokenSequence from "./components/TokenSequence";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/generate";

const DEFAULT_PROMPT =
  "Once upon a time, in a quaint village nestled between rolling hills, there lived";

export default function App() {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [steps, setSteps] = useState(50);
  const [genLength, setGenLength] = useState(100);
  const [blockLength, setBlockLength] = useState(100);

  const { status, tokens, step, totalSteps, error, start } = useDiffusionStream(WS_URL);

  const isBusy = status === "connecting" || status === "streaming";
  const maskedCount = tokens.filter((t) => t.masked).length;
  const progress = totalSteps > 0 ? step / totalSteps : 0;

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || isBusy) return;
    start({
      prompt: prompt.trim(),
      steps,
      gen_length: genLength,
      block_length: blockLength,
      temperature: 0,
      remasking: "low_confidence",
    });
  };

  return (
    <div className="min-h-screen bg-bg text-ink">
      <div className="mx-auto max-w-2xl px-6 py-12">
        <header className="mb-8">
          <h1 className="text-xl font-semibold">Diffusion LM Visualizer</h1>
          <p className="mt-1 text-sm text-muted">
            Watching LLaDA-8B-Instruct unmask a full sequence in parallel, one refinement step at
            a time.
          </p>
        </header>

        <form onSubmit={handleSubmit} className="rounded-lg border border-border bg-surface p-5">
          <label htmlFor="prompt" className="block text-sm text-muted">
            Prompt
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            className="mt-2 w-full resize-none rounded-md border border-border bg-bg p-3 font-sans text-sm text-ink outline-none focus:border-accent"
            placeholder="Once upon a time..."
          />

          <div className="mt-4 grid grid-cols-3 gap-3">
            <NumberField label="Steps" value={steps} onChange={setSteps} />
            <NumberField label="Length" value={genLength} onChange={setGenLength} />
            <NumberField label="Block" value={blockLength} onChange={setBlockLength} />
          </div>

          <button
            type="submit"
            disabled={isBusy}
            className="mt-4 w-full rounded-md bg-accent py-2 text-sm font-medium text-bg transition-opacity disabled:opacity-40"
          >
            {isBusy ? "Generating…" : "Generate"}
          </button>
        </form>

        <section className="mt-6 rounded-lg border border-border bg-surface p-5">
          <div className="mb-3 flex items-center justify-between text-sm text-muted">
            <span>
              {status === "idle" && "Waiting for a prompt"}
              {status === "connecting" && "Connecting…"}
              {status === "streaming" && `Step ${step} / ${totalSteps}`}
              {status === "complete" && `Done — ${totalSteps} steps`}
              {status === "error" && "Something went wrong"}
            </span>
            {tokens.length > 0 && status !== "error" && (
              <span>
                {maskedCount} of {tokens.length} still masked
              </span>
            )}
          </div>

          {(status === "streaming" || status === "complete") && (
            <div className="mb-4 h-1 w-full overflow-hidden rounded-full bg-bg">
              <div
                className="h-full bg-accent transition-[width] duration-300 ease-out"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          )}

          {status === "error" && error && <p className="mb-4 text-sm text-red-400">{error}</p>}

          <TokenSequence tokens={tokens} />
        </section>
      </div>
    </div>
  );
}

type NumberFieldProps = {
  label: string;
  value: number;
  onChange: (value: number) => void;
};

function NumberField({ label, value, onChange }: NumberFieldProps) {
  return (
    <label className="block text-sm text-muted">
      {label}
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full rounded-md border border-border bg-bg p-2 font-mono text-sm text-ink outline-none focus:border-accent"
      />
    </label>
  );
}
