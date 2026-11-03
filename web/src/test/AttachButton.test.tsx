/**
 * Tests for components/AttachButton.tsx.
 */
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import AttachButton from "../components/AttachButton";

afterEach(() => cleanup());

describe("AttachButton", () => {
  it("renders the attach button with an accessible label", () => {
    render(<AttachButton onFile={() => {}} disabled={false} progress={null} />);
    expect(
      screen.getByRole("button", { name: /attach a document/i }),
    ).toBeInTheDocument();
  });

  it("opens the hidden file picker on click", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <AttachButton onFile={() => {}} disabled={false} progress={null} />,
    );
    const input = container.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    const clickSpy = vi.spyOn(input, "click");
    await user.click(screen.getByRole("button", { name: /attach/i }));
    expect(clickSpy).toHaveBeenCalled();
  });

  it("calls onFile with the selected file", async () => {
    const user = userEvent.setup();
    const onFile = vi.fn();
    render(<AttachButton onFile={onFile} disabled={false} progress={null} />);
    const file = new File(["hello"], "doc.pdf", { type: "application/pdf" });
    const input = screen
      .getByRole("button", { name: /attach/i })
      .parentElement!.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);
    expect(onFile).toHaveBeenCalledTimes(1);
    expect(onFile.mock.calls[0][0].name).toBe("doc.pdf");
  });

  it("is disabled when the disabled prop is true", () => {
    render(<AttachButton onFile={() => {}} disabled={true} progress={null} />);
    expect(screen.getByRole("button", { name: /attach/i })).toBeDisabled();
  });

  it("shows the progress bar and uploading label while uploading", () => {
    const { container } = render(
      <AttachButton onFile={() => {}} disabled={false} progress={42} />,
    );
    expect(
      screen.getByRole("button", { name: /uploading document… 42%/i }),
    ).toBeInTheDocument();
    const bar = container.querySelector(".attach-progress") as HTMLElement;
    expect(bar.style.width).toBe("42%");
  });
});
