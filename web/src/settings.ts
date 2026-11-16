/**
 * GUI-only settings, persisted in browser localStorage. No backend involvement:
 * these control how the chat looks (avatars, theme, density, font size).
 */

export const SETTINGS_KEY = "sidekick-gui-settings";

export const THEME_OPTIONS = ["system", "light", "dark"] as const;
export const DENSITY_OPTIONS = ["comfortable", "compact"] as const;
export const FONT_SIZE_OPTIONS = ["small", "medium", "large"] as const;
export const SIDEBAR_ICON_IDS = ["spark", "bot", "chat"] as const;

export type ThemeSetting = (typeof THEME_OPTIONS)[number];
export type DensitySetting = (typeof DENSITY_OPTIONS)[number];
export type FontSizeSetting = (typeof FONT_SIZE_OPTIONS)[number];
export type SidekickIcon = (typeof SIDEBAR_ICON_IDS)[number];

export interface AvatarColorPreset {
  id: string;
  label: string;
  /** CSS background value (a var() reference or a hex color). */
  background: string;
  /** CSS text color for the icon/initial on top of the background. */
  color: string;
}

export const USER_AVATAR_COLORS: AvatarColorPreset[] = [
  { id: "gray", label: "Gray", background: "var(--pill-bg)", color: "var(--text)" },
  { id: "blue", label: "Blue", background: "#2456c8", color: "#ffffff" },
  { id: "green", label: "Green", background: "#1e7e34", color: "#ffffff" },
  { id: "purple", label: "Purple", background: "#6b4fa1", color: "#ffffff" },
  { id: "orange", label: "Orange", background: "#b45309", color: "#ffffff" },
];

export const ASSISTANT_AVATAR_COLORS: AvatarColorPreset[] = [
  { id: "blue", label: "Blue", background: "var(--accent)", color: "#ffffff" },
  { id: "teal", label: "Teal", background: "#0e7c7b", color: "#ffffff" },
  { id: "green", label: "Green", background: "#1e7e34", color: "#ffffff" },
  { id: "purple", label: "Purple", background: "#6b4fa1", color: "#ffffff" },
  { id: "orange", label: "Orange", background: "#b45309", color: "#ffffff" },
];

export interface GuiSettings {
  displayName: string;
  /** Custom user avatar initial (up to 2 chars). Empty means derive from displayName. */
  userInitial: string;
  userColor: string;
  sidekickIcon: SidekickIcon;
  sidekickColor: string;
  showAvatars: boolean;
  theme: ThemeSetting;
  density: DensitySetting;
  fontSize: FontSizeSetting;
}

export const DEFAULT_SETTINGS: GuiSettings = {
  displayName: "Bharath",
  userInitial: "",
  userColor: "gray",
  sidekickIcon: "spark",
  sidekickColor: "blue",
  showAvatars: true,
  theme: "system",
  density: "comfortable",
  fontSize: "medium",
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function pickString(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function pickBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function pickEnum<T extends string>(
  value: unknown,
  options: readonly T[],
  fallback: T,
): T {
  if (typeof value === "string") {
    for (const option of options) {
      if (option === value) return option;
    }
  }
  return fallback;
}

/** Read settings from localStorage, falling back per-field on anything invalid. */
export function loadSettings(): GuiSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) return { ...DEFAULT_SETTINGS };
    return {
      displayName: pickString(parsed.displayName, DEFAULT_SETTINGS.displayName),
      userInitial: pickString(parsed.userInitial, DEFAULT_SETTINGS.userInitial),
      userColor: pickString(parsed.userColor, DEFAULT_SETTINGS.userColor),
      sidekickIcon: pickEnum(
        parsed.sidekickIcon,
        SIDEBAR_ICON_IDS,
        DEFAULT_SETTINGS.sidekickIcon,
      ),
      sidekickColor: pickString(
        parsed.sidekickColor,
        DEFAULT_SETTINGS.sidekickColor,
      ),
      showAvatars: pickBoolean(parsed.showAvatars, DEFAULT_SETTINGS.showAvatars),
      theme: pickEnum(parsed.theme, THEME_OPTIONS, DEFAULT_SETTINGS.theme),
      density: pickEnum(parsed.density, DENSITY_OPTIONS, DEFAULT_SETTINGS.density),
      fontSize: pickEnum(parsed.fontSize, FONT_SIZE_OPTIONS, DEFAULT_SETTINGS.fontSize),
    };
  } catch {
    // Corrupt JSON or storage unavailable — defaults keep the GUI working.
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(settings: GuiSettings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    // Storage unavailable — the settings still apply for this session.
  }
}

/** Clear stored settings and return the defaults. */
export function resetSettings(): GuiSettings {
  try {
    localStorage.removeItem(SETTINGS_KEY);
  } catch {
    // Nothing to clear.
  }
  return { ...DEFAULT_SETTINGS };
}

/** Apply theme, density, and font size to the document root. */
export function applySettings(settings: GuiSettings): void {
  const root = document.documentElement;
  if (settings.theme === "system") {
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", settings.theme);
  }
  root.setAttribute("data-density", settings.density);
  root.setAttribute("data-font-size", settings.fontSize);
}

/** Find a color preset by id, falling back to the first preset on unknown ids. */
export function resolveAvatarColor(
  presets: AvatarColorPreset[],
  id: string,
): AvatarColorPreset {
  return presets.find((p) => p.id === id) ?? presets[0];
}

/** The initial shown in the user avatar: custom value, else the display name's first letter. */
export function resolveUserInitial(settings: GuiSettings): string {
  const custom = settings.userInitial.trim();
  if (custom) return custom.slice(0, 2).toUpperCase();
  const name = settings.displayName.trim();
  if (name) return name.charAt(0).toUpperCase();
  return "B";
}
