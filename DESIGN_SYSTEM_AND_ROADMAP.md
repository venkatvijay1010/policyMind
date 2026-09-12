# PolicyMind Product Design System and Roadmap

## Scope and product assessment

PolicyMind currently has one customer-facing surface: a local policy-intelligence
workspace with document upload, scoped retrieval, conversation, citations, and
generated SQL details. It does not currently have separate dashboard, analytics,
people, order, partner, or settings pages. This roadmap improves the real product
surface and describes the next pages worth building; it does not invent unrelated
admin screens.

| Dimension | Before redesign | Current direction | Public-launch target |
| --- | ---: | ---: | ---: |
| UX clarity | 6.0/10 | 7.8/10 | 9.5/10 |
| Visual quality | 6.5/10 | 8.0/10 | 9.5/10 |
| Accessibility | 7.0/10 | 8.4/10 | 9.5/10 |
| Mobile experience | 5.5/10 | 7.5/10 | 9.0/10 |
| Enterprise readiness | 5.5/10 | 7.5/10 | 9.5/10 |

The previous screen had a strong visual foundation but behaved like a polished
developer demo: the hero consumed too much of the workspace, provenance was
repeated as plain text, actions were hidden in the sidebar, and deletion used a
browser-native confirmation. The updated workspace prioritizes the question,
the supporting evidence, and rapid keyboard control.

## Product principles

1. Evidence is a first-class product object. Every policy answer should make its
   source, section, page or passage, and excerpt easy to inspect.
2. Local operation is a trust signal, not visual decoration. Show it succinctly
   and never let it compete with the user’s work.
3. One clear primary action per state. Ask, add a source, or inspect evidence.
4. Dense enough for daily work, quiet enough for focused reading.
5. Motion confirms cause and effect; it should never delay a task.

## Implemented system

### Foundations

| Token family | Values / use |
| --- | --- |
| Canvas | `#0b1017` deep neutral canvas |
| Surface | `#111925`, `#162131`, `#1b2a3d` for layered work areas |
| Content | `#f2f7fc`, `#9eb0c2`, `#70859a` for primary, secondary, tertiary text |
| Action | `#69d7c7` with `#8aebdc` hover state |
| Status | Info `#89b9ff`, critical `#f19a90` |
| Radius | 8, 12, 16, and 22 px |
| Elevation | restrained 1 px border plus small and medium shadows |
| Motion | 160–300 ms `cubic-bezier(.2,.8,.2,1)`; reduced-motion override included |

Typography stays on the system UI stack for fast, privacy-preserving local
delivery. The hierarchy is intentionally compact: 36 px maximum product title,
15 px body copy, 12–13 px controls, and 10–11 px metadata/eyebrows.

### Interaction and information architecture

- The compact workspace breadcrumb and `Answers you can verify` headline replace
  generic marketing copy.
- `Ctrl/Cmd + K` opens a command center with new conversation, focus composer,
  upload, document scope, and suggested-question actions.
- The command center supports search, arrow-key selection, Enter to execute, and
  Escape to dismiss.
- The welcome state provides two deliberate next steps rather than an empty chat.
- Upload and deletion give in-product toast feedback; deletion uses an accessible
  modal confirmation instead of `window.confirm`.
- The mobile layout preserves new conversation and command access while removing
  low-priority sidebar density.

### Evidence experience

Evidence cards are the sole detailed source presentation. They expose a section
and a physical PDF page when the source provides one. Text and Markdown sources
show a passage number rather than a fabricated page number. The older duplicate
plain-text source block has been removed.

## Accessibility standard

The implemented baseline includes semantic buttons, labelled dialogs, live
regions for notifications, a skip link, visible keyboard focus, keyboard command
navigation, keyboard dialog dismissal, and a reduced-motion mode. The visual
system is designed around high-contrast text on neutral dark surfaces.

Before a formal launch, validate it with automated axe checks and manual testing
using NVDA/Chrome, VoiceOver/Safari, keyboard-only navigation, 200% zoom, and a
320 px viewport. The acceptance target is WCAG 2.2 AA with no critical or serious
automated violations.

## Recommended next screens

These are the highest-value additions for an enterprise policy-intelligence
product, in priority order:

1. **Document detail** — source metadata, extraction status, version history,
   page preview, section list, and re-index control.
2. **Conversation history** — named saved conversations, folders, retention,
   sharing controls, and audit metadata.
3. **Source quality** — ingest errors, chunk and embedding coverage, citation
   quality, and re-index recommendations.
4. **Workspace settings** — model connection, allowed source hosts, retention,
   role-based access, and data export controls.
5. **Evaluation center** — retrieval precision, answer quality, latency, and
   regression comparison. This should be an evaluation workflow, not a generic
   vanity dashboard.

## Prioritized delivery plan

### P0: next launch increment

- Add browser-level visual regression checks for desktop and mobile.
- Add automated accessibility tests and a manual keyboard/screen-reader test
  script.
- Persist conversation history with clear delete/export controls.
- Add a document detail route and PDF source-page preview.

### P1: enterprise operating model

- Introduce authentication, organizations, roles, and an audit trail.
- Add document versions, source freshness, background indexing progress, and
  retryable ingest failures.
- Add evaluated answer feedback and retrieval-quality metrics.

### P2: scale and polish

- Replace in-process SQLite similarity ranking with an indexed vector backend
  when corpus size demands it.
- Add command-palette recents and personalized shortcuts.
- Offer a light theme only after it passes the same contrast and visual QA bar as
  the dark workspace.

## Definition of done for a 9.5/10 launch

- A new user can upload a policy, ask a question, understand the answer route,
  and inspect a source page without documentation.
- Every destructive action has a clear consequence and reversible path where
  feasible.
- No critical accessibility issues; keyboard operation is complete.
- Mobile access supports the core ask, source selection, upload, evidence review,
  and conversation reset flows.
- Observability measures time-to-first-answer, retrieval failures, ingest
  failures, answer feedback, and citation interaction rate.
