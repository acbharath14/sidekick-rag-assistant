import { useEffect } from "react";
import Avatar, { AssistantIcon } from "./Avatar";
import {
  ASSISTANT_AVATAR_COLORS,
  DENSITY_OPTIONS,
  FONT_SIZE_OPTIONS,
  THEME_OPTIONS,
  USER_AVATAR_COLORS,
} from "../settings";
import type {
  DensitySetting,
  FontSizeSetting,
  GuiSettings,
  SidekickIcon,
  ThemeSetting,
} from "../settings";

interface Props {
  settings: GuiSettings;
  onChange: (next: GuiSettings) => void;
  onReset: () => void;
  onClose: () => void;
}

const SIDEBAR_ICON_CHOICES: { id: SidekickIcon; label: string }[] = [
  { id: "spark", label: "Spark" },
  { id: "bot", label: "Bot" },
  { id: "chat", label: "Chat bubble" },
];

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export default function SettingsPanel({
  settings,
  onChange,
  onReset,
  onClose,
}: Props) {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent): void => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const handleReset = (): void => {
    if (!window.confirm("Reset all settings to defaults?")) return;
    onReset();
  };

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div
        className="settings-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="settings-header">
          <h2>Settings</h2>
          <button
            type="button"
            className="settings-close"
            onClick={onClose}
            aria-label="Close settings"
            title="Close settings"
          >
            ✕
          </button>
        </div>

        <section className="settings-section">
          <h3>Your profile</h3>
          <div className="settings-preview">
            <Avatar role="user" settings={{ ...settings, showAvatars: true }} />
            <span className="settings-preview-name">
              {settings.displayName || "You"}
            </span>
          </div>
          <label className="settings-field">
            <span>Display name</span>
            <input
              type="text"
              value={settings.displayName}
              onChange={(e) =>
                onChange({ ...settings, displayName: e.target.value })
              }
              aria-label="Display name"
            />
          </label>
          <label className="settings-field">
            <span>
              Avatar initial (up to 2 characters; leave blank to use your
              display name)
            </span>
            <input
              type="text"
              value={settings.userInitial}
              maxLength={2}
              onChange={(e) =>
                onChange({ ...settings, userInitial: e.target.value })
              }
              aria-label="Avatar initial"
            />
          </label>
          <div className="settings-field">
            <span>Avatar color</span>
            <div className="swatches">
              {USER_AVATAR_COLORS.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  className={
                    "swatch" +
                    (settings.userColor === c.id ? " swatch-selected" : "")
                  }
                  style={{ background: c.background }}
                  onClick={() => onChange({ ...settings, userColor: c.id })}
                  title={c.label}
                  aria-label={`User avatar color: ${c.label}`}
                  aria-pressed={settings.userColor === c.id}
                />
              ))}
            </div>
          </div>
        </section>

        <section className="settings-section">
          <h3>Sidekick</h3>
          <div className="settings-preview">
            <Avatar
              role="assistant"
              settings={{ ...settings, showAvatars: true }}
            />
            <span className="settings-preview-name">Sidekick</span>
          </div>
          <div className="settings-field">
            <span>Avatar icon</span>
            <div className="icon-choices">
              {SIDEBAR_ICON_CHOICES.map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  className={
                    "icon-choice" +
                    (settings.sidekickIcon === opt.id
                      ? " icon-choice-selected"
                      : "")
                  }
                  onClick={() =>
                    onChange({ ...settings, sidekickIcon: opt.id })
                  }
                  title={opt.label}
                  aria-label={`Sidekick icon: ${opt.label}`}
                  aria-pressed={settings.sidekickIcon === opt.id}
                >
                  <AssistantIcon icon={opt.id} />
                </button>
              ))}
            </div>
          </div>
          <div className="settings-field">
            <span>Avatar color</span>
            <div className="swatches">
              {ASSISTANT_AVATAR_COLORS.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  className={
                    "swatch" +
                    (settings.sidekickColor === c.id ? " swatch-selected" : "")
                  }
                  style={{ background: c.background }}
                  onClick={() =>
                    onChange({ ...settings, sidekickColor: c.id })
                  }
                  title={c.label}
                  aria-label={`Sidekick avatar color: ${c.label}`}
                  aria-pressed={settings.sidekickColor === c.id}
                />
              ))}
            </div>
          </div>
        </section>

        <section className="settings-section">
          <h3>Appearance</h3>
          <div className="settings-field">
            <span>Theme</span>
            <div className="segmented" role="group" aria-label="Theme">
              {THEME_OPTIONS.map((t: ThemeSetting) => (
                <button
                  key={t}
                  type="button"
                  className={
                    "segmented-btn" +
                    (settings.theme === t ? " segmented-btn-active" : "")
                  }
                  aria-pressed={settings.theme === t}
                  onClick={() => onChange({ ...settings, theme: t })}
                >
                  {capitalize(t)}
                </button>
              ))}
            </div>
          </div>
          <div className="settings-field">
            <span>Message density</span>
            <div
              className="segmented"
              role="group"
              aria-label="Message density"
            >
              {DENSITY_OPTIONS.map((d: DensitySetting) => (
                <button
                  key={d}
                  type="button"
                  className={
                    "segmented-btn" +
                    (settings.density === d ? " segmented-btn-active" : "")
                  }
                  aria-pressed={settings.density === d}
                  onClick={() => onChange({ ...settings, density: d })}
                >
                  {capitalize(d)}
                </button>
              ))}
            </div>
          </div>
          <div className="settings-field">
            <span>Font size</span>
            <div className="segmented" role="group" aria-label="Font size">
              {FONT_SIZE_OPTIONS.map((f: FontSizeSetting) => (
                <button
                  key={f}
                  type="button"
                  className={
                    "segmented-btn" +
                    (settings.fontSize === f ? " segmented-btn-active" : "")
                  }
                  aria-pressed={settings.fontSize === f}
                  onClick={() => onChange({ ...settings, fontSize: f })}
                >
                  {capitalize(f)}
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="settings-section">
          <h3>Chat display</h3>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.showAvatars}
              onChange={(e) =>
                onChange({ ...settings, showAvatars: e.target.checked })
              }
            />
            <span>Show avatars</span>
          </label>
        </section>

        <div className="settings-actions">
          <button
            type="button"
            className="settings-reset"
            onClick={handleReset}
          >
            Reset to defaults
          </button>
          <button type="button" className="settings-done" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
