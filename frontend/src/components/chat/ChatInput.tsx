'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  DropdownMenu,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
} from '@astryxdesign/core/DropdownMenu';
import { Tooltip } from '@astryxdesign/core/Tooltip';
import { ArrowUp, ChevronDown, Globe, Mic, Paperclip, TriangleAlert, Upload, X } from 'lucide-react';
import { AttachmentItem } from '@/types/chat';
import { haptics } from '@/utils/haptics';
import { AttachmentIcon, formatFileSize } from './AttachmentIcon';

type SearchMode = 'auto' | 'on' | 'off';

const SEARCH_MODE_OPTIONS: { value: SearchMode; label: string; description: string }[] = [
  { value: 'auto', label: 'Auto', description: 'Search only when the question needs fresh information' },
  { value: 'on', label: 'Always', description: 'Every message searches the web' },
  { value: 'off', label: 'Off', description: 'Never search the web' },
];
const SEARCH_CHIP_TEXT: Record<SearchMode, string> = { auto: 'Search', on: 'Search on', off: 'Search off' };
// Sent to the API: true = always, false = never, undefined = let the server decide.
const SEARCH_MODE_FLAG: Record<SearchMode, boolean | undefined> = { auto: undefined, on: true, off: false };

const MAX_FILE_BYTES = 15 * 1024 * 1024;
const NOTICE_TIMEOUT_MS = 7000;
const HERO_MAX_HEIGHT = 220;
const DOCKED_MAX_HEIGHT = 160;

interface ChatInputProps {
  onSend: (message: string, attachments?: AttachmentItem[], webSearch?: boolean) => void;
  onStop: () => void;
  isStreaming: boolean;
  isHero?: boolean;
  initialValue?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  onStop,
  isStreaming,
  isHero = false,
  initialValue = '',
}) => {
  const [input, setInput] = useState(initialValue);
  const [attachments, setAttachments] = useState<AttachmentItem[]>([]);
  // Auto lets the server decide; Always and Off are sent explicitly (true / false).
  const [searchMode, setSearchMode] = useState<SearchMode>('auto');
  const [isRecording, setIsRecording] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const recognitionRef = useRef<any>(null);
  const attachmentsRef = useRef<AttachmentItem[]>([]);


  useEffect(() => {
    attachmentsRef.current = attachments;
  }, [attachments]);

  useEffect(() => {
    if (initialValue) {
      setInput(initialValue);
      textareaRef.current?.focus();
    }
  }, [initialValue]);

  // Listen for image edit requests from GeneratedImageCard
  useEffect(() => {
    const handleEditImage = (e: Event) => {
      const customEvent = e as CustomEvent<{ prompt: string; imageUrl?: string }>;
      if (customEvent.detail?.prompt) {
        setInput(customEvent.detail.prompt);
        setTimeout(() => {
          textareaRef.current?.focus();
          textareaRef.current?.select();
        }, 50);
      }
    };
    window.addEventListener('jru:edit-image', handleEditImage);
    return () => window.removeEventListener('jru:edit-image', handleEditImage);
  }, []);

  // Release preview URLs and the microphone only when the composer unmounts.
  // (Revoking on every change broke the previews of files still attached.)
  useEffect(() => {
    return () => {
      attachmentsRef.current.forEach((att) => {
        if (att.previewUrl) URL.revokeObjectURL(att.previewUrl);
      });
      try {
        recognitionRef.current?.stop();
      } catch {
        // Recognition already stopped
      }
    };
  }, []);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(null), NOTICE_TIMEOUT_MS);
    return () => clearTimeout(timer);
  }, [notice]);

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const maxH = isHero ? HERO_MAX_HEIGHT : DOCKED_MAX_HEIGHT;
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, maxH)}px`;
    }
  }, [input, isHero]);

  const updateInput = (value: string) => {
    setInput(value);
  };

  // File processing to Base64
  const processFiles = useCallback((files: FileList | File[]) => {
    const fileArray = Array.from(files);
    const tooLarge = fileArray.filter((file) => file.size > MAX_FILE_BYTES);
    if (tooLarge.length > 0) {
      setNotice(`${tooLarge.map((f) => f.name).join(', ')} ${tooLarge.length === 1 ? 'is' : 'are'} over the 15 MB limit.`);
    }
    fileArray
      .filter((file) => file.size <= MAX_FILE_BYTES)
      .forEach((file) => {
        const reader = new FileReader();
        reader.onload = () => {
          const resultStr = reader.result as string;
          const base64Data = resultStr.includes(',') ? resultStr.split(',')[1] : resultStr;
          const isImage = file.type.startsWith('image/');
          const newAttachment: AttachmentItem = {
            name: file.name,
            mime_type: file.type || 'application/octet-stream',
            data: base64Data,
            size: file.size,
            previewUrl: isImage ? URL.createObjectURL(file) : undefined,
          };
          setAttachments((prev) => [...prev, newAttachment]);
        };
        reader.onerror = () => setNotice(`Could not read ${file.name}. Try attaching it again.`);
        reader.readAsDataURL(file);
      });
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files);
      e.target.value = '';
    }
  };

  const removeAttachment = (index: number) => {
    setAttachments((prev) => {
      const item = prev[index];
      if (item?.previewUrl) URL.revokeObjectURL(item.previewUrl);
      return prev.filter((_, i) => i !== index);
    });
    textareaRef.current?.focus();
  };

  // Drag & drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files);
    }
  };

  const handleSubmit = useCallback(() => {
    const trimmed = input.trim();
    if ((trimmed || attachments.length > 0) && !isStreaming) {
      haptics.send();
      const messageContent = trimmed || 'Analyze attached file(s)';
      onSend(messageContent, attachments.length > 0 ? attachments : undefined, SEARCH_MODE_FLAG[searchMode]);
      setInput('');
      setAttachments([]);
      setNotice(null);
      if (textareaRef.current) textareaRef.current.style.height = 'auto';
    }
  }, [input, attachments, isStreaming, searchMode, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Never act on Enter while an input method (Malayalam, Hindi, ...) is composing.
    // keyCode 229 covers Safari, which reports the composition-ending Enter after isComposing turns false.
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const [isTranscribing, setIsTranscribing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  // Release the microphone on unmount without uploading a half-finished recording.
  useEffect(() => {
    return () => {
      const recorder = mediaRecorderRef.current;
      if (recorder) {
        recorder.ondataavailable = null;
        recorder.onstop = null;
        if (recorder.state !== 'inactive') {
          try {
            recorder.stop();
          } catch {}
        }
        mediaRecorderRef.current = null;
      }
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    };
  }, []);

  // Cloud Voice Input (Groq Whisper) or Fallback Web Speech API.
  // On by default; the server decides availability, and a refusal switches this session to browser dictation.
  const [isCloudVoiceUnavailable, setIsCloudVoiceUnavailable] = useState(false);
  const isCloudVoiceEnabled =
    process.env.NEXT_PUBLIC_ENABLE_VOICE_TRANSCRIPTION !== 'false' && !isCloudVoiceUnavailable;

  const toggleRecording = async () => {
    if (isRecording) {
      if (isCloudVoiceEnabled && mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
        return;
      }
      try {
        recognitionRef.current?.stop();
      } catch {
        // Recognition already stopped
      }
      setIsRecording(false);
      return;
    }

    if (isCloudVoiceEnabled && typeof navigator !== 'undefined' && navigator.mediaDevices?.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaStreamRef.current = stream;
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) audioChunksRef.current.push(event.data);
        };

        mediaRecorder.onstop = async () => {
          stream.getTracks().forEach((track) => track.stop());
          if (mediaStreamRef.current === stream) mediaStreamRef.current = null;
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          if (audioBlob.size === 0) return;

          setIsTranscribing(true);
          try {
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.webm');
            const authToken = localStorage.getItem('jru_auth_token') || localStorage.getItem('token');
            const resp = await fetch('/api/v1/audio/transcribe', {
              method: 'POST',
              credentials: 'include',
              headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
              body: formData,
            });
            if (resp.ok) {
              const data = await resp.json();
              if (data.text) setInput((prev) => (prev ? `${prev.trim()} ${data.text}` : data.text));
            } else {
              if (resp.status === 503 || resp.status === 429) setIsCloudVoiceUnavailable(true);
              setNotice('Voice input is unavailable right now. Tap the mic again to use your browser’s dictation.');
            }
          } catch {
            setNotice('Could not upload your recording. Check your connection and try again.');
          } finally {
            setIsTranscribing(false);
          }
        };

        mediaRecorder.start();
        setIsRecording(true);
        return;
      } catch {
        // Microphone blocked or MediaRecorder missing: fall back to browser dictation below
      }
    }

    const SpeechRecognition =
      typeof window !== 'undefined' && ((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition);

    if (!SpeechRecognition) {
      setNotice('Voice input is not supported in this browser. Try Chrome, Edge or Safari.');
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      recognition.onstart = () => setIsRecording(true);
      recognition.onresult = (event: any) => {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) finalTranscript += event.results[i][0].transcript + ' ';
        }
        if (finalTranscript) setInput((prev) => (prev ? `${prev.trim()} ${finalTranscript}` : finalTranscript));
      };
      recognition.onerror = () => {
        setIsRecording(false);
        setNotice('Voice input stopped. Check that Jruva may use your microphone.');
      };
      recognition.onend = () => setIsRecording(false);
      recognitionRef.current = recognition;
      recognition.start();
    } catch {
      setIsRecording(false);
      setNotice('Could not start voice input. Check that Jruva may use your microphone.');
    }
  };

  const hasContent = Boolean(input.trim() || attachments.length > 0);
  const micLabel = isTranscribing ? 'Transcribing…' : isRecording ? 'Stop voice input' : 'Voice input';
  const placeholder = isRecording ? 'Listening…' : isHero ? 'Ask Jruva anything' : 'Reply to Jruva';

  return (
    <div className={isHero ? 'jru-hero-input-wrap' : 'jru-docked-input-wrap'}>
      <div
        className={`jru-input-card ${isHero ? 'hero' : 'docked'} ${isDragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={handleFileChange}
          multiple
          accept="image/*,.pdf,.txt,.md,.json,.py,.js,.ts,.tsx,.csv"
        />

        {isDragging && (
          <div className="jru-drag-overlay">
            <Upload className="jru-drag-icon" size={22} strokeWidth={1.75} aria-hidden="true" />
            <span>Drop images, PDFs or documents here</span>
          </div>
        )}

        {attachments.length > 0 && (
          <ul className="jru-attachments-preview-row" aria-label="Attached files">
            {attachments.map((att, idx) => (
              <li key={`${att.name}-${idx}`} className="jru-attachment-chip">
                {att.previewUrl ? (
                  <img src={att.previewUrl} alt="" className="jru-chip-thumbnail" />
                ) : (
                  <div className="jru-chip-doc-icon">
                    <AttachmentIcon name={att.name} mimeType={att.mime_type} size={16} />
                  </div>
                )}
                <div className="jru-chip-meta">
                  <span className="jru-chip-filename" title={att.name}>
                    {att.name}
                  </span>
                  {att.size ? <span className="jru-chip-size">{formatFileSize(att.size)}</span> : null}
                </div>
                <button
                  type="button"
                  className="jru-chip-remove-btn"
                  onClick={() => removeAttachment(idx)}
                  aria-label={`Remove ${att.name}`}
                >
                  <X size={13} strokeWidth={2.25} aria-hidden="true" />
                </button>
              </li>
            ))}
          </ul>
        )}

        <textarea
          ref={textareaRef}
          className="jru-textarea"
          rows={isHero ? 2 : 1}
          placeholder={placeholder}
          value={input}
          onChange={(e) => updateInput(e.target.value)}
          onKeyDown={handleKeyDown}
          aria-label="Message"
        />

        {notice && (
          <div className="jru-input-notice" role="alert">
            <TriangleAlert size={14} strokeWidth={2} aria-hidden="true" />
            <span>{notice}</span>
            <button type="button" className="jru-input-notice-close" onClick={() => setNotice(null)} aria-label="Dismiss">
              <X size={13} strokeWidth={2.25} aria-hidden="true" />
            </button>
          </div>
        )}

        <div className="jru-input-action-bar">
          <div className="jru-input-left-tools">
            <Tooltip content="Attach images, PDFs or code" placement="above">
              <button
                type="button"
                className="jru-circle-tool-btn"
                onClick={() => fileInputRef.current?.click()}
                aria-label="Attach files"
              >
                <Paperclip size={15} strokeWidth={2} aria-hidden="true" />
              </button>
            </Tooltip>

            <DropdownMenu
              menuWidth={320}
              placement="above"
              alignment="start"
              renderTrigger={(triggerProps) => (
                <button
                  type="button"
                  {...triggerProps}
                  className={`jru-websearch-tool-btn ${searchMode === 'on' ? 'active' : ''} ${searchMode === 'off' ? 'off' : ''}`}
                  aria-label={`Web search: ${SEARCH_MODE_OPTIONS.find((o) => o.value === searchMode)?.label}`}
                >
                  <Globe size={14} strokeWidth={2} aria-hidden="true" />
                  <span className="jru-search-btn-text">{SEARCH_CHIP_TEXT[searchMode]}</span>
                  <ChevronDown size={12} strokeWidth={2.25} aria-hidden="true" />
                </button>
              )}
            >
              <DropdownMenuRadioGroup
                label="Web search"
                value={searchMode}
                onChange={(value) => setSearchMode(value as SearchMode)}
              >
                {SEARCH_MODE_OPTIONS.map((option) => (
                  <DropdownMenuRadioItem
                    key={option.value}
                    value={option.value}
                    label={option.label}
                    description={option.description}
                  />
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenu>
          </div>

          <div className="jru-input-right-tools">
            <Tooltip content={micLabel} placement="above">
              <button
                type="button"
                onClick={toggleRecording}
                disabled={isTranscribing}
                className={`jru-mic-btn ${isRecording ? 'recording' : ''} ${isTranscribing ? 'transcribing' : ''}`}
                aria-label={micLabel}
                aria-pressed={isRecording}
              >
                {isRecording ? <span className="jru-mic-pulse-ring" /> : null}
                <Mic size={15} strokeWidth={2} aria-hidden="true" />
              </button>
            </Tooltip>

            {isStreaming ? (
              <Tooltip content="Stop answer (Esc)" placement="above">
                <button type="button" onClick={onStop} className="jru-circle-send-btn stop" aria-label="Stop answer">
                  <div className="stop-square" />
                </button>
              </Tooltip>
            ) : (
              <button
                type="button"
                onClick={handleSubmit}
                disabled={!hasContent}
                className={`jru-circle-send-btn ${hasContent ? 'active' : ''}`}
                aria-label="Send message"
              >
                <ArrowUp size={16} strokeWidth={2.4} aria-hidden="true" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
