/**
 * Tests for components/MicButton.tsx, with window.SpeechRecognition mocked.
 */
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import MicButton from "../components/MicButton";

type ResultEvent = {
  resultIndex: number;
  results: Array<{
    isFinal: boolean;
    0: { transcript: string };
    length: number;
  }>;
};

class MockRecognition {
  static instances: MockRecognition[] = [];
  continuous = false;
  interimResults = false;
  lang = "";
  onresult: ((ev: ResultEvent) => void) | null = null;
  onerror: ((ev: { error: string }) => void) | null = null;
  onend: (() => void) | null = null;
  start = vi.fn();
  stop = vi.fn(() => {
    this.onend?.();
  });
  abort = vi.fn();

  constructor() {
    MockRecognition.instances.push(this);
  }
}

function installMock(): void {
  (window as unknown as Record<string, unknown>).SpeechRecognition =
    MockRecognition;
  delete (window as unknown as Record<string, unknown>)
    .webkitSpeechRecognition;
}

function removeMock(): void {
  delete (window as unknown as Record<string, unknown>).SpeechRecognition;
  delete (window as unknown as Record<string, unknown>)
    .webkitSpeechRecognition;
}

function resultEvent(
  transcript: string,
  isFinal: boolean,
): ResultEvent {
  return {
    resultIndex: 0,
    results: [
      { isFinal, 0: { transcript }, length: 1 },
    ],
  };
}

function lastInstance(): MockRecognition {
  const instances = MockRecognition.instances;
  if (instances.length === 0) throw new Error("no recognition created");
  return instances[instances.length - 1];
}

beforeEach(() => {
  MockRecognition.instances = [];
  installMock();
});

afterEach(() => {
  removeMock();
  cleanup();
});

function renderMic(props?: {
  value?: string;
  onChange?: (value: string) => void;
  onDictationError?: (message: string) => void;
  disabled?: boolean;
}) {
  const onChange = props?.onChange ?? vi.fn();
  const onDictationError = props?.onDictationError ?? vi.fn();
  render(
    <MicButton
      disabled={props?.disabled ?? false}
      value={props?.value ?? ""}
      onChange={onChange}
      onDictationError={onDictationError}
    />,
  );
  return { onChange, onDictationError };
}

describe("MicButton", () => {
  it("renders the mic button with an accessible label when supported", () => {
    renderMic();
    expect(
      screen.getByRole("button", { name: /dictate with voice input/i }),
    ).toBeInTheDocument();
  });

  it("is disabled with an explanatory title when the API is unsupported", () => {
    removeMock();
    renderMic();
    const button = screen.getByRole("button", {
      name: /dictate with voice input/i,
    });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute(
      "title",
      expect.stringContaining("not supported in this browser"),
    );
  });

  it("starts listening on click with continuous=false, interim results, en-US", async () => {
    const user = userEvent.setup();
    renderMic();
    await user.click(
      screen.getByRole("button", { name: /dictate with voice input/i }),
    );
    const recognition = lastInstance();
    expect(recognition.start).toHaveBeenCalledTimes(1);
    expect(recognition.continuous).toBe(false);
    expect(recognition.interimResults).toBe(true);
    expect(recognition.lang).toBe("en-US");
    const button = screen.getByRole("button", { name: /stop voice input/i });
    expect(button).toHaveAttribute("aria-pressed", "true");
    expect(button.className).toContain("mic-button-listening");
  });

  it("shows the interim transcript live in the input", async () => {
    const user = userEvent.setup();
    const { onChange } = renderMic({ value: "" });
    await user.click(
      screen.getByRole("button", { name: /dictate with voice input/i }),
    );
    lastInstance().onresult?.(resultEvent("hello wo", false));
    expect(onChange).toHaveBeenCalledWith("hello wo");
  });

  it("appends the final transcript to existing text with a space", async () => {
    const user = userEvent.setup();
    const { onChange } = renderMic({ value: "ask about" });
    await user.click(
      screen.getByRole("button", { name: /dictate with voice input/i }),
    );
    lastInstance().onresult?.(resultEvent("the refund policy", true));
    expect(onChange).toHaveBeenCalledWith("ask about the refund policy");
  });

  it("does not duplicate the interim text when recognition ends", async () => {
    const user = userEvent.setup();
    const { onChange } = renderMic({ value: "" });
    await user.click(
      screen.getByRole("button", { name: /dictate with voice input/i }),
    );
    const recognition = lastInstance();
    recognition.onresult?.(resultEvent("hello", false));
    recognition.onend?.();
    const calls = (onChange as ReturnType<typeof vi.fn>).mock.calls.map(
      (c: unknown[]) => c[0],
    );
    expect(calls[calls.length - 1]).toBe("hello");
    expect(calls.filter((c: unknown) => c === "hello hello")).toHaveLength(0);
  });

  it("stops listening on a second click", async () => {
    const user = userEvent.setup();
    renderMic();
    const button = screen.getByRole("button", {
      name: /dictate with voice input/i,
    });
    await user.click(button);
    const recognition = lastInstance();
    await user.click(screen.getByRole("button", { name: /stop voice input/i }));
    expect(recognition.stop).toHaveBeenCalledTimes(1);
    expect(
      screen.getByRole("button", { name: /dictate with voice input/i }),
    ).toHaveAttribute("aria-pressed", "false");
  });

  it("reports microphone permission denial", async () => {
    const user = userEvent.setup();
    const { onDictationError } = renderMic();
    await user.click(
      screen.getByRole("button", { name: /dictate with voice input/i }),
    );
    lastInstance().onerror?.({ error: "not-allowed" });
    expect(onDictationError).toHaveBeenCalledWith(
      expect.stringContaining("Microphone access denied"),
    );
  });

  it("stays quiet on no-speech instead of showing an error", async () => {
    const user = userEvent.setup();
    const { onDictationError } = renderMic();
    await user.click(
      screen.getByRole("button", { name: /dictate with voice input/i }),
    );
    lastInstance().onerror?.({ error: "no-speech" });
    expect(onDictationError).not.toHaveBeenCalled();
  });

  it("is disabled when the disabled prop is true", () => {
    renderMic({ disabled: true });
    expect(
      screen.getByRole("button", { name: /dictate with voice input/i }),
    ).toBeDisabled();
  });
});
