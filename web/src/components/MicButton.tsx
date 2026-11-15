import { useEffect, useRef, useState } from "react";

/**
 * Minimal Web Speech API types. TypeScript's DOM lib ships the result
 * types (SpeechRecognitionResultList) but not the recognizer itself or
 * its events, so those are declared here.
 */
interface DictationResultEvent extends Event {
  readonly resultIndex: number;
  readonly results: SpeechRecognitionResultList;
}

interface DictationErrorEvent extends Event {
  readonly error: string;
}

interface DictationRecognition {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((this: DictationRecognition, ev: DictationResultEvent) => void) | null;
  onerror: ((this: DictationRecognition, ev: DictationErrorEvent) => void) | null;
  onend: ((this: DictationRecognition, ev: Event) => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

type DictationRecognitionConstructor = new () => DictationRecognition;

interface DictationWindow {
  SpeechRecognition?: DictationRecognitionConstructor;
  webkitSpeechRecognition?: DictationRecognitionConstructor;
}

function getRecognitionConstructor(): DictationRecognitionConstructor | null {
  const w = window as Window & DictationWindow;
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

function joinText(base: string, addition: string): string {
  if (!addition) return base;
  if (!base) return addition;
  return `${base} ${addition}`;
}

function stripInterim(value: string, interim: string): string {
  if (!interim) return value;
  if (value === interim) return "";
  const suffix = ` ${interim}`;
  if (value.endsWith(suffix)) return value.slice(0, -suffix.length);
  return value;
}

interface Props {
  disabled: boolean;
  value: string;
  onChange: (value: string) => void;
  onDictationError: (message: string) => void;
}

export default function MicButton({ disabled, value, onChange, onDictationError }: Props) {
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef<DictationRecognition | null>(null);
  const valueRef = useRef(value);
  const interimRef = useRef("");

  useEffect(() => {
    valueRef.current = value;
  });

  useEffect(() => {
    return () => {
      const recognition = recognitionRef.current;
      recognitionRef.current = null;
      if (recognition) {
        recognition.onresult = null;
        recognition.onerror = null;
        recognition.onend = null;
        recognition.abort();
      }
    };
  }, []);

  const supported = getRecognitionConstructor() !== null;

  const toggle = (): void => {
    if (listening) {
      recognitionRef.current?.stop();
      return;
    }
    const Ctor = getRecognitionConstructor();
    if (!Ctor) return;
    const recognition = new Ctor();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.onresult = (event) => {
      let interim = "";
      let finalText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const text = result[0]?.transcript ?? "";
        if (result.isFinal) {
          finalText += text;
        } else {
          interim += text;
        }
      }
      const committed = stripInterim(valueRef.current, interimRef.current);
      if (finalText) {
        interimRef.current = "";
        onChange(joinText(committed, finalText.trim()));
      } else if (interim) {
        interimRef.current = interim;
        onChange(joinText(committed, interim));
      }
    };
    recognition.onerror = (event) => {
      if (event.error === "no-speech") return;
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        onDictationError(
          "Microphone access denied. Allow microphone permission for this site and try again.",
        );
      } else {
        onDictationError("Voice input failed. Check your microphone and try again.");
      }
    };
    recognition.onend = () => {
      setListening(false);
      recognitionRef.current = null;
      const pending = interimRef.current;
      if (pending) {
        interimRef.current = "";
        const committed = stripInterim(valueRef.current, pending);
        onChange(joinText(committed, pending.trim()));
      }
    };
    recognitionRef.current = recognition;
    try {
      recognition.start();
      setListening(true);
    } catch {
      recognitionRef.current = null;
      onDictationError("Voice input failed. Check your microphone and try again.");
    }
  };

  return (
    <button
      type="button"
      className={"mic-button" + (listening ? " mic-button-listening" : "")}
      onClick={toggle}
      disabled={disabled || !supported}
      aria-label={listening ? "Stop voice input" : "Dictate with voice input"}
      aria-pressed={listening}
      title={
        !supported
          ? "Voice input is not supported in this browser (try Chrome or Edge)"
          : listening
            ? "Stop listening"
            : "Dictate with voice input"
      }
    >
      <span aria-hidden="true">🎙️</span>
    </button>
  );
}
