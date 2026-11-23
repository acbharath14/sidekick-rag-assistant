/**
 * Tests for the GUI settings: localStorage persistence, Avatar variants
 * driven by settings, and the settings panel.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Avatar from "../components/Avatar";
import ChatMessage from "../components/ChatMessage";
import SettingsPanel from "../components/SettingsPanel";
import {
  DEFAULT_SETTINGS,
  applySettings,
  loadSettings,
  resetSettings,
  saveSettings,
} from "../settings";
import type { ChatResponse } from "../types";

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
  document.documentElement.removeAttribute("data-density");
  document.documentElement.removeAttribute("data-font-size");
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("settings persistence", () => {
  it("returns defaults when nothing is stored", () => {
    expect(loadSettings()).toEqual(DEFAULT_SETTINGS);
  });

  it("round-trips through localStorage", () => {
    const next = {
      ...DEFAULT_SETTINGS,
      displayName: "Alice",
      theme: "dark" as const,
      showAvatars: false,
    };
    saveSettings(next);
    expect(loadSettings()).toEqual(next);
  });

  it("falls back to defaults on corrupt JSON", () => {
    localStorage.setItem("sidekick-gui-settings", "{not json");
    expect(loadSettings()).toEqual(DEFAULT_SETTINGS);
  });

  it("falls back per-field on invalid values, keeping the valid ones", () => {
    localStorage.setItem(
      "sidekick-gui-settings",
      JSON.stringify({
        ...DEFAULT_SETTINGS,
        theme: "neon",
        fontSize: "huge",
        displayName: "Alice",
      }),
    );
    const loaded = loadSettings();
    expect(loaded.theme).toBe("system");
    expect(loaded.fontSize).toBe("medium");
    expect(loaded.displayName).toBe("Alice");
  });

  it("resetSettings clears the stored key", () => {
    saveSettings({ ...DEFAULT_SETTINGS, theme: "dark" });
    expect(resetSettings()).toEqual(DEFAULT_SETTINGS);
    expect(localStorage.getItem("sidekick-gui-settings")).toBeNull();
  });
});

describe("applySettings", () => {
  it("sets data-theme for light and dark, removes it for system", () => {
    applySettings({ ...DEFAULT_SETTINGS, theme: "dark" });
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    applySettings({ ...DEFAULT_SETTINGS, theme: "light" });
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    applySettings({ ...DEFAULT_SETTINGS, theme: "system" });
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
  });

  it("sets density and font-size attributes", () => {
    applySettings({
      ...DEFAULT_SETTINGS,
      density: "compact",
      fontSize: "large",
    });
    expect(document.documentElement.getAttribute("data-density")).toBe(
      "compact",
    );
    expect(document.documentElement.getAttribute("data-font-size")).toBe(
      "large",
    );
  });
});

describe("Avatar with settings", () => {
  it("uses a custom initial", () => {
    render(
      <Avatar
        role="user"
        settings={{ ...DEFAULT_SETTINGS, userInitial: "ak" }}
      />,
    );
    expect(screen.getByRole("img", { name: "You" })).toHaveTextContent("AK");
  });

  it("derives the initial from the display name when no custom initial is set", () => {
    render(
      <Avatar
        role="user"
        settings={{
          ...DEFAULT_SETTINGS,
          displayName: "Alice",
          userInitial: "",
        }}
      />,
    );
    expect(screen.getByRole("img", { name: "You" })).toHaveTextContent("A");
  });

  it("renders the bot and chat Sidekick icons", () => {
    const { rerender } = render(
      <Avatar
        role="assistant"
        settings={{ ...DEFAULT_SETTINGS, sidekickIcon: "bot" }}
      />,
    );
    expect(
      screen.getByRole("img", { name: "Sidekick" }).getAttribute("data-icon"),
    ).toBe("bot");
    rerender(
      <Avatar
        role="assistant"
        settings={{ ...DEFAULT_SETTINGS, sidekickIcon: "chat" }}
      />,
    );
    expect(
      screen.getByRole("img", { name: "Sidekick" }).getAttribute("data-icon"),
    ).toBe("chat");
  });

  it("applies a custom avatar color", () => {
    render(
      <Avatar
        role="user"
        settings={{ ...DEFAULT_SETTINGS, userColor: "green" }}
      />,
    );
    const avatar = screen.getByRole("img", { name: "You" });
    expect(avatar.style.background).toMatch(/1e7e34|30, 126, 52/);
  });

  it("renders nothing when avatars are off", () => {
    const { container } = render(
      <Avatar
        role="user"
        settings={{ ...DEFAULT_SETTINGS, showAvatars: false }}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("hides the Sidekick avatar in ChatMessage when avatars are off", () => {
    const response: ChatResponse = {
      text: "Hi.",
      sources: [],
      hits: [],
      grounded: true,
      latency_ms: 10,
      standalone_question: null,
    };
    render(
      <ChatMessage
        question="Hi"
        response={response}
        settings={{ ...DEFAULT_SETTINGS, showAvatars: false }}
      />,
    );
    expect(screen.queryByRole("img", { name: "Sidekick" })).toBeNull();
    expect(screen.getByText("Hi.")).toBeInTheDocument();
  });
});

describe("SettingsPanel", () => {
  it("renders every section and updates the display name", () => {
    const onChange = vi.fn();
    render(
      <SettingsPanel
        settings={DEFAULT_SETTINGS}
        onChange={onChange}
        onReset={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    expect(
      screen.getByRole("dialog", { name: "Settings" }),
    ).toBeInTheDocument();
    for (const heading of [
      "Your profile",
      "Sidekick",
      "Appearance",
      "Chat display",
    ]) {
      expect(
        screen.getByRole("heading", { name: heading }),
      ).toBeInTheDocument();
    }
    fireEvent.change(screen.getByLabelText("Display name"), {
      target: { value: "Alice" },
    });
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ displayName: "Alice" }),
    );
  });

  it("reset asks for confirmation, then calls onReset", () => {
    const onReset = vi.fn();
    vi.stubGlobal("confirm", () => true);
    render(
      <SettingsPanel
        settings={DEFAULT_SETTINGS}
        onChange={vi.fn()}
        onReset={onReset}
        onClose={vi.fn()}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Reset to defaults" }),
    );
    expect(onReset).toHaveBeenCalled();
  });

  it("does not reset when the confirmation is declined", () => {
    const onReset = vi.fn();
    vi.stubGlobal("confirm", () => false);
    render(
      <SettingsPanel
        settings={DEFAULT_SETTINGS}
        onChange={vi.fn()}
        onReset={onReset}
        onClose={vi.fn()}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Reset to defaults" }),
    );
    expect(onReset).not.toHaveBeenCalled();
  });

  it("closes on Escape", () => {
    const onClose = vi.fn();
    render(
      <SettingsPanel
        settings={DEFAULT_SETTINGS}
        onChange={vi.fn()}
        onReset={vi.fn()}
        onClose={onClose}
      />,
    );
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });
});
