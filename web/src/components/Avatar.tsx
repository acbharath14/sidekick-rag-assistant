interface Props {
  role: "user" | "assistant";
}

/**
 * Speaker avatar shown to the left of each chat message, so it is obvious
 * who said what: a spark mark in an accent circle for Sidekick, the
 * initial "B" in a neutral circle for the user.
 */
export default function Avatar({ role }: Props) {
  if (role === "assistant") {
    return (
      <span
        className="avatar avatar-assistant"
        role="img"
        aria-label="Sidekick"
        title="Sidekick"
      >
        <svg
          viewBox="0 0 24 24"
          width="18"
          height="18"
          fill="currentColor"
          aria-hidden="true"
        >
          <path d="M12 2l2.1 6.9L21 11l-6.9 2.1L12 20l-2.1-6.9L3 11l6.9-2.1z" />
        </svg>
      </span>
    );
  }
  return (
    <span
      className="avatar avatar-user"
      role="img"
      aria-label="You"
      title="You"
    >
      B
    </span>
  );
}
