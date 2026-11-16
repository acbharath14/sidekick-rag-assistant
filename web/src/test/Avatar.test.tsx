/**
 * Tests for components/Avatar.tsx: one distinct avatar per speaker.
 */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import Avatar from "../components/Avatar";

afterEach(() => {
  cleanup();
});

describe("Avatar", () => {
  it("renders the Sidekick avatar as an accent circle with a spark mark", () => {
    const { container } = render(<Avatar role="assistant" />);
    const avatar = screen.getByRole("img", { name: "Sidekick" });
    expect(avatar).toBeInTheDocument();
    expect(avatar.className).toContain("avatar-assistant");
    expect(container.querySelector("svg")).not.toBeNull();
  });

  it("renders the user avatar as a circle with the initial B", () => {
    render(<Avatar role="user" />);
    const avatar = screen.getByRole("img", { name: "You" });
    expect(avatar).toBeInTheDocument();
    expect(avatar.className).toContain("avatar-user");
    expect(avatar).toHaveTextContent("B");
  });

  it("gives each speaker a visually distinct avatar", () => {
    const { rerender } = render(<Avatar role="assistant" />);
    const assistantClass = screen.getByRole("img", { name: "Sidekick" })
      .className;
    rerender(<Avatar role="user" />);
    const userClass = screen.getByRole("img", { name: "You" }).className;
    expect(assistantClass).not.toBe(userClass);
    expect(assistantClass).toContain("avatar-assistant");
    expect(userClass).toContain("avatar-user");
  });
});
