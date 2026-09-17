import React from "react";

type Props = {
  children: React.ReactNode;
};

type State = {
  error: Error | null;
};

export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Keep details in console for debugging.
    // eslint-disable-next-line no-console
    console.error("Unhandled React render error:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 24, fontFamily: "ui-sans-serif, system-ui" }}>
          <h1 style={{ fontSize: 18, fontWeight: 700, marginBottom: 12 }}>
            App crashed while rendering
          </h1>
          <p style={{ marginBottom: 12 }}>
            Open DevTools Console for full stack trace.
          </p>
          <pre
            style={{
              background: "#111827",
              color: "#e5e7eb",
              padding: 12,
              borderRadius: 8,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {this.state.error.message}
          </pre>
        </div>
      );
    }

    return this.props.children;
  }
}

