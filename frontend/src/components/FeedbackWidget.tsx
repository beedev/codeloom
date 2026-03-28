/**
 * FeedbackWidget -- inline feedback UI attached to assistant messages.
 *
 * Ported from dbn-v2 with adaptations:
 *   - notebookId -> projectId
 *   - submitFeedback -> submitChatFeedback
 *
 * Flow:
 *   1. Thumbs up / thumbs down buttons (always visible, one click to expand)
 *   2. After thumbs selection -> expand panel with star rating, category chips, optional note
 *   3. Submit -> POST /api/projects/{id}/chat/feedback -> show confirmation
 */

import { memo, useState, useCallback } from 'react';
import { ThumbsUp, ThumbsDown, Star, X, Send, Check } from 'lucide-react';
import { submitChatFeedback } from '../services/api.ts';

interface FeedbackWidgetProps {
  traceId: string;
  projectId: string;
}

type FeedbackCategory = 'helpful' | 'inaccurate' | 'irrelevant' | 'incomplete' | 'other';
type SubmitState = 'idle' | 'submitting' | 'submitted' | 'error';

const CATEGORIES: { value: FeedbackCategory; label: string }[] = [
  { value: 'helpful', label: 'Helpful' },
  { value: 'inaccurate', label: 'Inaccurate' },
  { value: 'irrelevant', label: 'Irrelevant' },
  { value: 'incomplete', label: 'Incomplete' },
  { value: 'other', label: 'Other' },
];

export const FeedbackWidget = memo(function FeedbackWidget({
  traceId,
  projectId,
}: FeedbackWidgetProps) {
  const [helpful, setHelpful] = useState<boolean | null>(null);
  const [rating, setRating] = useState<number | null>(null);
  const [hoverRating, setHoverRating] = useState<number | null>(null);
  const [category, setCategory] = useState<FeedbackCategory | null>(null);
  const [userMessage, setUserMessage] = useState('');
  const [expanded, setExpanded] = useState(false);
  const [submitState, setSubmitState] = useState<SubmitState>('idle');
  const [errorMessage, setErrorMessage] = useState('');

  const handleThumbsClick = useCallback((value: boolean) => {
    if (submitState === 'submitted') return;
    setHelpful(value);
    setExpanded(true);
  }, [submitState]);

  const handleSubmit = useCallback(async () => {
    if (submitState !== 'idle') return;
    if (helpful === null && rating === null) {
      setErrorMessage('Please select a rating or helpful indicator.');
      return;
    }

    setSubmitState('submitting');
    setErrorMessage('');

    try {
      await submitChatFeedback(
        projectId,
        traceId,
        helpful ?? true,
        category ?? undefined,
        userMessage.trim() || undefined,
      );
      setSubmitState('submitted');
    } catch (err) {
      console.error('[FeedbackWidget] Failed to submit feedback:', err);
      setSubmitState('error');
      setErrorMessage('Could not save feedback. Please try again.');
    }
  }, [submitState, helpful, rating, traceId, projectId, category, userMessage]);

  const handleDismiss = useCallback(() => {
    setExpanded(false);
    if (submitState !== 'submitted') {
      setHelpful(null);
    }
  }, [submitState]);

  if (submitState === 'submitted') {
    return (
      <div className="flex items-center gap-1.5 mt-3 text-xs text-emerald-400">
        <Check className="w-3.5 h-3.5" />
        <span>Thanks for your feedback</span>
      </div>
    );
  }

  return (
    <div className="mt-3">
      {/* Thumbs row */}
      <div className="flex items-center gap-1">
        <span className="text-xs text-text-dim mr-1">Helpful?</span>
        <button
          onClick={() => handleThumbsClick(true)}
          disabled={submitState === 'submitting'}
          title="Thumbs up"
          className={`
            p-1.5 rounded-lg transition-all duration-150
            ${helpful === true
              ? 'bg-emerald-400/20 text-emerald-400'
              : 'text-text-dim hover:text-emerald-400 hover:bg-emerald-400/10'
            }
          `}
        >
          <ThumbsUp className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={() => handleThumbsClick(false)}
          disabled={submitState === 'submitting'}
          title="Thumbs down"
          className={`
            p-1.5 rounded-lg transition-all duration-150
            ${helpful === false
              ? 'bg-red-400/20 text-red-400'
              : 'text-text-dim hover:text-red-400 hover:bg-red-400/10'
            }
          `}
        >
          <ThumbsDown className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Expanded panel */}
      {expanded && (
        <div className="mt-2 p-3 rounded-lg bg-[#1a233b] border border-[#2a3352] space-y-3">
          {/* Header */}
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-text-muted">Share more details (optional)</span>
            <button
              onClick={handleDismiss}
              className="p-0.5 rounded text-text-dim hover:text-text transition-colors"
              title="Dismiss"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Star rating */}
          <div className="flex items-center gap-1">
            <span className="text-xs text-text-dim mr-1">Rating:</span>
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                onClick={() => setRating(star === rating ? null : star)}
                onMouseEnter={() => setHoverRating(star)}
                onMouseLeave={() => setHoverRating(null)}
                title={`${star} star${star !== 1 ? 's' : ''}`}
                className="p-0.5 transition-colors"
              >
                <Star
                  className={`w-4 h-4 transition-colors ${
                    star <= (hoverRating ?? rating ?? 0)
                      ? 'fill-amber-400 text-amber-400'
                      : 'fill-transparent text-text-dim'
                  }`}
                />
              </button>
            ))}
          </div>

          {/* Category chips */}
          <div className="flex flex-wrap gap-1.5">
            {CATEGORIES.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => setCategory(category === value ? null : value)}
                className={`
                  px-2 py-0.5 rounded-full text-xs border transition-all duration-150
                  ${category === value
                    ? 'bg-blue-400/20 text-blue-400 border-blue-400/40'
                    : 'text-text-dim border-[#2a3352] hover:text-text hover:border-text-dim'
                  }
                `}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Note textarea */}
          <textarea
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            placeholder="Additional comments... (optional)"
            maxLength={500}
            rows={2}
            className="
              w-full px-2.5 py-1.5 rounded-lg text-xs
              bg-[#0f172a] border border-[#2a3352]
              text-white placeholder:text-text-dim
              focus:outline-none focus:border-blue-400/40
              resize-none transition-colors
            "
          />

          {/* Error */}
          {errorMessage && (
            <p className="text-xs text-red-400">{errorMessage}</p>
          )}

          {/* Submit */}
          <div className="flex justify-end">
            <button
              onClick={handleSubmit}
              disabled={submitState === 'submitting' || (helpful === null && rating === null)}
              className="
                inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs
                bg-blue-400/20 text-blue-400 border border-blue-400/30
                hover:bg-blue-400/30 transition-all duration-150
                disabled:opacity-40 disabled:cursor-not-allowed
              "
            >
              {submitState === 'submitting' ? (
                <>
                  <span className="w-3.5 h-3.5 rounded-full border-2 border-blue-400 border-t-transparent animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Send className="w-3.5 h-3.5" />
                  Submit Feedback
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
});

export default FeedbackWidget;
