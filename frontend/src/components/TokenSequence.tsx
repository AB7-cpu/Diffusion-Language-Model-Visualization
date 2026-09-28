import type { TokenState } from "../hooks/useDiffusionStream";

type Props = {
  tokens: TokenState[];
};

// Model special tokens (<|eot_id|>, <|endoftext|>, ...) are real output, not
// noise to hide — but they read as clutter sitting inline with prose, so
// they're muted rather than removed. The visualizer should never show
// anything that didn't actually come from the model.
function isSpecialToken(text: string): boolean {
  return text.startsWith("<") && text.endsWith(">");
}

export default function TokenSequence({ tokens }: Props) {
  if (tokens.length === 0) {
    return (
      <p className="font-mono text-sm text-muted">
        Masked positions will appear here once generation starts.
      </p>
    );
  }

  return (
    <p className="break-words font-mono text-[15px] leading-8">
      {tokens.map((token, i) =>
        token.masked ? (
          <span
            key={i}
            className="mx-[1px] inline-block h-4 w-4 align-middle rounded-sm border border-dashed border-border bg-masked"
            title="[MASK]"
          />
        ) : (
          <span
            key={i}
            className={
              "token-revealed " + (isSpecialToken(token.text) ? "text-muted" : "text-ink")
            }
            style={{ whiteSpace: "pre-wrap" }}
          >
            {token.text}
          </span>
        )
      )}
    </p>
  );
}
