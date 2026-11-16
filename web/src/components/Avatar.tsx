import {
  ASSISTANT_AVATAR_COLORS,
  DEFAULT_SETTINGS,
  USER_AVATAR_COLORS,
  resolveAvatarColor,
  resolveUserInitial,
} from "../settings";
import type { GuiSettings, SidekickIcon } from "../settings";

interface Props {
  role: "user" | "assistant";
  settings?: GuiSettings;
}

/** The Sidekick avatar mark. Shared with the settings panel's icon picker. */
export function AssistantIcon({ icon }: { icon: SidekickIcon }) {
  if (icon === "bot") {
    return (
      <svg
        viewBox="0 0 24 24"
        width="18"
        height="18"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        aria-hidden="true"
      >
        <line x1="12" y1="3" x2="12" y2="6" />
        <circle cx="12" cy="2.4" r="1" fill="currentColor" stroke="none" />
        <rect x="4" y="6" width="16" height="12" rx="3" />
        <circle cx="9" cy="11" r="1.1" fill="currentColor" stroke="none" />
        <circle cx="15" cy="11" r="1.1" fill="currentColor" stroke="none" />
        <line x1="9" y1="15" x2="15" y2="15" />
      </svg>
    );
  }
  if (icon === "chat") {
    return (
      <svg
        viewBox="0 0 24 24"
        width="18"
        height="18"
        fill="currentColor"
        aria-hidden="true"
      >
        <path d="M12 3C6.48 3 2 6.94 2 11.75c0 2.72 1.46 5.11 3.74 6.6L5 21l3.5-1.75c1.1.3 2.28.5 3.5.5 5.52 0 10-3.94 10-8.75S17.52 3 12 3z" />
      </svg>
    );
  }
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M12 2l2.1 6.9L21 11l-6.9 2.1L12 20l-2.1-6.9L3 11l6.9-2.1z" />
    </svg>
  );
}

/**
 * Speaker avatar shown to the left of each chat message, so it is obvious
 * who said what. Appearance comes from the GUI settings (display name,
 * custom initial, icon, colors); with no settings the original look is kept:
 * a spark mark in an accent circle for Sidekick, the initial "B" in a
 * neutral circle for the user. Returns nothing when avatars are turned off.
 */
export default function Avatar({ role, settings = DEFAULT_SETTINGS }: Props) {
  if (!settings.showAvatars) return null;

  if (role === "assistant") {
    const preset = resolveAvatarColor(
      ASSISTANT_AVATAR_COLORS,
      settings.sidekickColor,
    );
    return (
      <span
        className="avatar avatar-assistant"
        role="img"
        aria-label="Sidekick"
        title="Sidekick"
        data-icon={settings.sidekickIcon}
        style={{ background: preset.background, color: preset.color }}
      >
        <AssistantIcon icon={settings.sidekickIcon} />
      </span>
    );
  }

  const preset = resolveAvatarColor(USER_AVATAR_COLORS, settings.userColor);
  return (
    <span
      className="avatar avatar-user"
      role="img"
      aria-label="You"
      title={settings.displayName || "You"}
      style={{ background: preset.background, color: preset.color }}
    >
      {resolveUserInitial(settings)}
    </span>
  );
}
