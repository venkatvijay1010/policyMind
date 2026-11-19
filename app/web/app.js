const form = document.querySelector("#chat-form");
const promptInput = document.querySelector("#prompt");
const scopeInput = document.querySelector("#scope-key");
const retrievalSelect = document.querySelector("#retrieval-strategy");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#send-button");
const sendLabel = sendButton.querySelector(".send-label");
const sendIcon = sendButton.querySelector(".send-icon");
const newChatButton = document.querySelector("#new-chat");
const providerChip = document.querySelector("#provider-chip");
const localStatus = document.querySelector("#local-status");
const scopeSummary = document.querySelector("#scope-summary");
const scopeList = document.querySelector("#scope-list");
const uploadToggle = document.querySelector("#upload-toggle");
const uploadForm = document.querySelector("#upload-form");
const uploadTitle = document.querySelector("#upload-title");
const uploadFile = document.querySelector("#upload-file");
const uploadButton = document.querySelector("#upload-button");
const uploadStatus = document.querySelector("#upload-status");
const selectedFile = document.querySelector("#selected-file");
const dropzone = document.querySelector("#dropzone");
const composerFeedback = document.querySelector("#composer-feedback");
const toastRegion = document.querySelector("#toast-region");
const commandTrigger = document.querySelector("#command-trigger");
const commandPalette = document.querySelector("#command-palette");
const commandClose = document.querySelector("#command-close");
const commandInput = document.querySelector("#command-input");
const commandResults = document.querySelector("#command-results");
const confirmDialog = document.querySelector("#confirm-dialog");
const confirmCopy = document.querySelector("#confirm-copy");
const themeToggle = document.querySelector("#theme-toggle");
const sourceViewer = document.querySelector("#source-viewer");
const sourceViewerTitle = document.querySelector("#source-viewer-title");
const sourceViewerSubtitle = document.querySelector("#source-viewer-subtitle");
const sourceViewerOriginal = document.querySelector("#source-viewer-original");
const sourceViewerClose = document.querySelector("#source-viewer-close");
const sourceViewerPager = document.querySelector("#source-viewer-pager");
const sourceViewerPrev = document.querySelector("#source-viewer-prev");
const sourceViewerNext = document.querySelector("#source-viewer-next");
const sourceViewerPage = document.querySelector("#source-viewer-page");
const sourceViewerBody = document.querySelector("#source-viewer-body");
const sourceViewerStatus = document.querySelector("#source-viewer-status");

let isSending = false;
let activeController = null;
let knownScopes = [];
let uploadProgressTimer = null;
let conversationHistory = [];
let commandItems = [];
let selectedCommandIndex = 0;
let activeSource = null;
let sourceRenderToken = 0;

const SOURCES_API = "/api/v2/knowledge/sources";

const THEME_STORAGE_KEY = "policymind-theme";

function preferredTheme() {
  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (storedTheme === "light" || storedTheme === "dark") return storedTheme;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function applyTheme(theme, persist = true) {
  const nextTheme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = nextTheme;
  themeToggle.setAttribute("aria-pressed", String(nextTheme === "light"));
  themeToggle.setAttribute("aria-label", `Switch to ${nextTheme === "light" ? "dark" : "light"} theme`);
  themeToggle.querySelector(".theme-icon").innerHTML = nextTheme === "light" ? "&#9728;" : "&#9789;";
  themeToggle.querySelector(".theme-label").textContent = nextTheme === "light" ? "Light" : "Dark";
  if (persist) window.localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
}

const documentStageOrder = ["understanding", "retrieving", "evidence", "drafting"];
const chatStageOrder = ["understanding", "chatting"];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatAnswer(value) {
  const escaped = escapeHtml(value || "");
  return escaped
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br />");
}

function formatDuration(milliseconds) {
  const seconds = Number(milliseconds || 0) / 1000;
  if (seconds < 1) return `${Math.round(milliseconds || 0)} ms`;
  return `${seconds.toFixed(seconds >= 10 ? 1 : 2)}s`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function scrollToLatest() {
  requestAnimationFrame(() => {
    messages.scrollTo({ top: messages.scrollHeight, behavior: "smooth" });
  });
}

function resizeComposer() {
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 160)}px`;
}

function setComposerFeedback(message = "", state = "") {
  composerFeedback.textContent = message;
  composerFeedback.dataset.state = state;
}

function showToast(message, tone = "success") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${tone}`;
  toast.innerHTML = `<span class="toast-icon" aria-hidden="true">${tone === "success" ? "&#10003;" : "!"}</span><span>${escapeHtml(message)}</span>`;
  toastRegion.append(toast);
  window.setTimeout(() => toast.classList.add("is-leaving"), 3200);
  window.setTimeout(() => toast.remove(), 3500);
}

function commandCatalog() {
  return [
    { type: "new", title: "New conversation", detail: "Clear this conversation and start fresh", icon: "&#10022;" },
    { type: "ask", title: "Ask a question", detail: "Focus the policy intelligence composer", icon: "&#8599;" },
    { type: "upload", title: "Add a document", detail: "Upload a TXT, Markdown, or PDF source", icon: "&#8679;" },
    { type: "theme", title: "Switch theme", detail: "Toggle between light and dark appearance", icon: "&#9788;" },
    ...knownScopes.map((scope) => ({
      type: "scope",
      scopeKey: String(scope.scope_key),
      title: scope.title || "Untitled document",
      detail: `${scope.segment_count} indexed passages`,
      icon: "&#9636;",
    })),
    ...Array.from(document.querySelectorAll(".suggestion")).map((button) => ({
      type: "prompt",
      prompt: button.dataset.prompt || "",
      title: button.textContent?.trim() || "Suggested question",
      detail: "Run this example", icon: "&#10022;",
    })),
  ];
}

function renderCommandResults(query = "") {
  const normalized = query.trim().toLowerCase();
  commandItems = commandCatalog().filter((item) =>
    `${item.title} ${item.detail}`.toLowerCase().includes(normalized),
  );
  selectedCommandIndex = 0;
  commandResults.innerHTML = commandItems.length
    ? commandItems
        .map((item, index) => `
          <button class="command-result" type="button" role="option" aria-selected="false" data-command-index="${index}">
            <span class="command-result-icon" aria-hidden="true">${item.icon}</span>
            <span><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.detail)}</small></span>
            <span class="command-result-arrow" aria-hidden="true">&rarr;</span>
          </button>
        `)
        .join("")
    : `<p class="command-empty">No matching actions or documents.</p>`;
  setCommandSelection(0);
}

function setCommandSelection(index) {
  if (!commandItems.length) return;
  selectedCommandIndex = (index + commandItems.length) % commandItems.length;
  commandResults.querySelectorAll(".command-result").forEach((button, buttonIndex) => {
    const isSelected = buttonIndex === selectedCommandIndex;
    button.classList.toggle("is-selected", isSelected);
    button.setAttribute("aria-selected", String(isSelected));
    if (isSelected) button.scrollIntoView({ block: "nearest" });
  });
}

function closeCommandPalette() {
  if (commandPalette?.open) commandPalette.close();
}

function openCommandPalette() {
  if (!commandPalette?.showModal) return;
  renderCommandResults();
  commandPalette.showModal();
  window.setTimeout(() => commandInput.focus(), 0);
}

function activateCommand(item) {
  if (!item) return;
  closeCommandPalette();
  if (item.type === "new") {
    newChatButton.click();
  } else if (item.type === "ask") {
    promptInput.focus();
  } else if (item.type === "upload") {
    toggleUpload(true);
  } else if (item.type === "theme") {
    applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
  } else if (item.type === "scope") {
    scopeInput.value = item.scopeKey;
    updateScopeSummary();
    promptInput.focus();
  } else if (item.type === "prompt") {
    promptInput.value = item.prompt;
    resizeComposer();
    updateSendButton();
    promptInput.focus();
  }
}

function requestDeleteConfirmation(documentTitle) {
  if (!confirmDialog?.showModal) {
    return Promise.resolve(window.confirm(`Delete "${documentTitle}" and its indexed passages?`));
  }
  confirmCopy.textContent = `This permanently removes “${documentTitle}” and all of its indexed passages.`;
  confirmDialog.showModal();
  return new Promise((resolve) => {
    confirmDialog.addEventListener("close", () => resolve(confirmDialog.returnValue === "confirm"), {
      once: true,
    });
  });
}

function updateSendButton() {
  if (isSending) {
    sendButton.disabled = false;
    return;
  }
  sendButton.disabled = !promptInput.value.trim();
}

function setSourceStatus(message = "", tone = "") {
  sourceViewerStatus.textContent = message;
  sourceViewerStatus.dataset.tone = tone;
}

function clearSourceImage() {
  if (activeSource?.imageUrl) URL.revokeObjectURL(activeSource.imageUrl);
  if (activeSource) activeSource.imageUrl = "";
}

function resetSourceViewer() {
  sourceRenderToken += 1;
  clearSourceImage();
  activeSource = null;
  sourceViewerBody.replaceChildren();
  sourceViewerPager.hidden = true;
  sourceViewerOriginal.hidden = true;
  setSourceStatus();
}

function sourceErrorMessage(body, fallback) {
  return typeof body?.detail === "string" && body.detail ? body.detail : fallback;
}

async function showPdfPage(page) {
  const source = activeSource;
  if (!source || source.kind !== "pdf") return;
  const target = source.pageCount > 0 ? Math.min(Math.max(1, page), source.pageCount) : Math.max(1, page);
  const token = ++sourceRenderToken;
  sourceViewerPrev.disabled = true;
  sourceViewerNext.disabled = true;
  setSourceStatus(`Loading page ${target}...`);

  try {
    const query = source.passageId ? `?passage_id=${encodeURIComponent(source.passageId)}` : "";
    const response = await fetch(`${SOURCES_API}/${encodeURIComponent(source.documentId)}/pages/${target}/image${query}`);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(sourceErrorMessage(body, "This page could not be displayed."));
    }
    const highlights = Number(response.headers.get("X-Highlight-Count")) || 0;
    const highlightTop = Number.parseFloat(response.headers.get("X-Highlight-Top"));
    const blob = await response.blob();
    if (token !== sourceRenderToken) return;

    const imageUrl = URL.createObjectURL(blob);
    const image = new Image();
    image.className = "viewer-page";
    image.alt = `Page ${target} of ${source.filename}${highlights ? ", with the cited paragraph highlighted" : ""}`;
    image.src = imageUrl;
    await image.decode().catch(() => {});
    if (token !== sourceRenderToken) {
      URL.revokeObjectURL(imageUrl);
      return;
    }

    clearSourceImage();
    source.imageUrl = imageUrl;
    source.page = target;
    sourceViewerBody.replaceChildren(image);
    sourceViewerBody.scrollTop = Number.isFinite(highlightTop)
      ? Math.max(0, image.offsetTop + image.clientHeight * highlightTop - sourceViewerBody.clientHeight * 0.25)
      : 0;
    sourceViewerPage.textContent = source.pageCount > 0 ? `Page ${target} of ${source.pageCount}` : `Page ${target}`;

    if (highlights) setSourceStatus("The cited paragraph is highlighted on this page.", "success");
    else if (source.passageId && target === source.citedPage) setSourceStatus("This is the cited page. The exact paragraph could not be marked.");
    else setSourceStatus();
  } catch (error) {
    if (token !== sourceRenderToken) return;
    setSourceStatus(error instanceof Error ? error.message : "This page could not be displayed.", "error");
  } finally {
    if (token === sourceRenderToken && activeSource === source) {
      sourceViewerPrev.disabled = source.page <= 1;
      sourceViewerNext.disabled = source.pageCount > 0 && source.page >= source.pageCount;
    }
  }
}

function renderTextSource(info) {
  const text = info.text || "";
  const pre = document.createElement("pre");
  pre.className = "viewer-text";
  let mark = null;
  if (info.highlighted && Number.isInteger(info.highlight_start) && Number.isInteger(info.highlight_end)) {
    mark = document.createElement("mark");
    mark.className = "viewer-mark";
    mark.textContent = text.slice(info.highlight_start, info.highlight_end);
    pre.append(text.slice(0, info.highlight_start), mark, text.slice(info.highlight_end));
  } else {
    pre.textContent = text;
  }
  sourceViewerBody.replaceChildren(pre);
  if (mark) sourceViewerBody.scrollTop = Math.max(0, mark.offsetTop - sourceViewerBody.clientHeight * 0.3);
  if (mark) setSourceStatus("The cited paragraph is highlighted.", "success");
  else setSourceStatus("The exact paragraph could not be marked in this document.");
}

async function openSource(documentId, passageId = "", title = "") {
  if (!sourceViewer?.showModal) {
    window.open(`${SOURCES_API}/${encodeURIComponent(documentId)}/file`, "_blank", "noopener");
    return;
  }
  resetSourceViewer();
  const token = ++sourceRenderToken;
  sourceViewerTitle.textContent = title || "Source document";
  sourceViewerSubtitle.textContent = "";
  setSourceStatus("Opening the source document...");
  if (!sourceViewer.open) sourceViewer.showModal();

  try {
    const query = passageId ? `?passage_id=${encodeURIComponent(passageId)}` : "";
    const response = await fetch(`${SOURCES_API}/${encodeURIComponent(documentId)}/open${query}`);
    const info = await response.json().catch(() => ({}));
    if (token !== sourceRenderToken) return;
    if (!response.ok) throw new Error(sourceErrorMessage(info, "Unable to open that source document."));

    const firstPage = info.page || 1;
    activeSource = {
      documentId: info.source_document_id,
      passageId: info.passage_id || "",
      kind: info.kind,
      filename: info.filename,
      pageCount: info.page_count || 0,
      citedPage: firstPage,
      page: firstPage,
      imageUrl: "",
    };
    sourceViewerTitle.textContent = title || info.filename;
    sourceViewerSubtitle.textContent = [info.section ? `Section: ${info.section}` : "", info.filename]
      .filter(Boolean)
      .join(" \u00b7 ");
    sourceViewerOriginal.href = info.kind === "pdf" ? `${info.file_url}#page=${firstPage}` : info.file_url;
    sourceViewerOriginal.hidden = false;

    if (info.kind === "pdf") {
      sourceViewerPager.hidden = false;
      await showPdfPage(firstPage);
    } else {
      renderTextSource(info);
    }
  } catch (error) {
    if (token !== sourceRenderToken) return;
    const message = error instanceof Error ? error.message : "Unable to open that source document.";
    setSourceStatus(message, "error");
    showToast(message, "error");
  }
}

function rememberConversationTurn(role, content) {
  const normalized = String(content || "").trim();
  if (!normalized) return;
  conversationHistory.push({ role, content: normalized.slice(0, 1600) });
  conversationHistory = conversationHistory.slice(-10);
}

function welcomeMarkup() {
  return `
    <article class="message assistant-message welcome-message">
      <div class="avatar" aria-hidden="true">P</div>
      <div class="message-content welcome-card">
        <p class="message-label"><span class="message-presence" aria-hidden="true"></span> PolicyMind</p>
        <div class="message-body">
          <p>Start with a policy question, then inspect the evidence behind every answer.</p>
          <p class="muted">Your local model decides whether to chat, retrieve policy passages, analyse data, or combine both.</p>
        </div>
        <div class="onboarding-actions">
          <button type="button" class="onboarding-action" data-quick-prompt="What benefit cap options are available?"><span aria-hidden="true">&#10022;</span> Explore a policy</button>
          <button type="button" class="onboarding-action" data-command="upload"><span aria-hidden="true">&#8679;</span> Add a document</button>
        </div>
      </div>
    </article>
  `;
}

function addUserMessage(prompt) {
  const node = document.createElement("article");
  node.className = "message user-message message-enter";
  node.innerHTML = `
    <div class="message-content">
      <p class="message-label">You</p>
      <div class="message-body"><p>${escapeHtml(prompt).replace(/\n/g, "<br />")}</p></div>
    </div>
    <div class="avatar user-avatar" aria-hidden="true">You</div>
  `;
  messages.append(node);
  scrollToLatest();
}

function updateLoadingMessage(element, status) {
  if (!element) return;
  const stage = status.stage || "understanding";
  const stageText = element.querySelector(".activity-stage");
  const detail = element.querySelector(".activity-detail");
  if (stageText) stageText.textContent = status.message || "Working locally";
  if (detail) {
    detail.textContent =
      stage === "chatting"
        ? "Using the local model to prepare a reply."
        : stage === "retrieving"
        ? "Looking through the information relevant to your request."
        : stage === "evidence"
          ? "Checking the passages that best support an answer."
          : stage === "drafting"
            ? "Generating a concise answer with Ollama on this computer."
            : "Understanding your message.";
  }

  const activePath = stage === "chatting" ? chatStageOrder : documentStageOrder;
  const stageIndex = activePath.indexOf(stage);
  element.querySelectorAll(".activity-steps span").forEach((step) => {
    const stepIndex = activePath.indexOf(step.dataset.stage);
    step.classList.toggle("is-active", step.dataset.stage === stage);
    step.classList.toggle("is-complete", stepIndex >= 0 && stageIndex > stepIndex);
  });
  scrollToLatest();
}

function addLoadingMessage() {
  const fragment = document.querySelector("#loading-template").content.cloneNode(true);
  const element = fragment.querySelector(".loading-message");
  const startedAt = performance.now();
  const elapsed = element.querySelector(".activity-elapsed");
  element._elapsedTimer = window.setInterval(() => {
    if (elapsed) elapsed.textContent = `${((performance.now() - startedAt) / 1000).toFixed(1)}s`;
  }, 100);
  updateLoadingMessage(element, { stage: "understanding", message: "Thinking" });
  messages.setAttribute("aria-busy", "true");
  messages.append(fragment);
  scrollToLatest();
  return element;
}

function removeLoadingMessage(element) {
  if (!element) return;
  if (element._elapsedTimer) window.clearInterval(element._elapsedTimer);
  element.remove();
  messages.setAttribute("aria-busy", "false");
}

function resultMeta(result) {
  const type = String(result.query_type || "document_qa").replaceAll("_", " ");
  const latency = Number.isFinite(result.latency_ms)
    ? `<span class="latency-chip">${formatDuration(result.latency_ms)} response time</span>`
    : "";
  return `<span class="query-type">${escapeHtml(type)}</span>${latency}`;
}

function citationActionMarkup(citation, index) {
  if (!citation.source_document_id) return "";
  const page = citation.page ? ` &middot; page ${escapeHtml(citation.page)}` : "";
  return `
    <div class="citation-actions">
      <button class="citation-open" type="button" data-source-document-id="${escapeHtml(citation.source_document_id)}" data-passage-id="${escapeHtml(citation.passage_id ?? "")}" data-title="${escapeHtml(citation.contract_title || "")}" aria-label="Open the source document for evidence ${index + 1}">Open source${page}</button>
    </div>`;
}

function citationsMarkup(citations) {
  if (!Array.isArray(citations) || citations.length === 0) return "";

  const items = citations
    .map(
      (citation, index) => `
        <details class="citation" open>
          <summary>
            <span class="citation-source"><b>${index + 1}</b> Evidence source</span>
            <span class="citation-document">${escapeHtml(citation.contract_title || "Policy document")}</span>
          </summary>
          <div class="citation-meta">
            <span class="citation-detail"><span aria-hidden="true">&sect;</span> Section: ${escapeHtml(citation.section || `Policy passage ${index + 1}`)}</span>
            <span class="citation-detail"><span aria-hidden="true">&#9636;</span> ${citation.page ? `Page ${escapeHtml(citation.page)}` : `Passage ${index + 1}`}</span>
          </div>
          <p class="citation-excerpt">${escapeHtml(citation.chunk_text || "")}</p>
          ${citationActionMarkup(citation, index)}
        </details>
      `,
    )
    .join("");

  return `<section class="sources"><div class="source-heading"><span>Evidence used</span><span>${citations.length} source${citations.length === 1 ? "" : "s"}</span></div>${items}</section>`;
}

function sqlMarkup(result) {
  if (!result.sql_query) return "";
  const rows = Array.isArray(result.sql_result) ? result.sql_result : [];
  return `
    <details class="sql-details">
      <summary>View generated SQLite query${rows.length ? ` | ${rows.length} row${rows.length === 1 ? "" : "s"}` : ""}</summary>
      <pre><code>${escapeHtml(result.sql_query)}</code></pre>
      ${rows.length ? `<pre class="sql-result"><code>${escapeHtml(JSON.stringify(rows, null, 2))}</code></pre>` : ""}
    </details>
  `;
}

function addAssistantMessage(result, errorMessage = "") {
  const node = document.createElement("article");
  node.className = "message assistant-message message-enter";
  const isError = Boolean(errorMessage);
  const answer = isError
    ? `<p class="error-text">${escapeHtml(errorMessage)}</p>`
    : formatAnswer(result.answer);

  node.innerHTML = `
    <div class="avatar" aria-hidden="true">P</div>
    <div class="message-content">
      <p class="message-label">PolicyMind</p>
      <div class="message-body ${isError ? "error-body" : ""}">${answer}</div>
      ${isError ? "" : `<div class="answer-meta">${resultMeta(result)}</div>`}
      ${isError ? "" : citationsMarkup(result.citations)}
      ${isError ? "" : sqlMarkup(result)}
    </div>
  `;
  messages.append(node);
  scrollToLatest();
}

function setSending(sending) {
  isSending = sending;
  promptInput.disabled = sending;
  sendButton.classList.toggle("is-cancel", sending);
  sendButton.setAttribute("aria-label", sending ? "Stop current request" : "Send question");
  sendLabel.textContent = sending ? "Stop" : "Send";
  sendIcon.textContent = sending ? "x" : "\u2191";
  updateSendButton();
}

async function consumeEventStream(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() || "";

    for (const frame of frames) {
      let event = "message";
      const dataLines = [];
      for (const line of frame.split(/\r?\n/)) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length) onEvent(event, JSON.parse(dataLines.join("\n")));
    }
    if (done) break;
  }
}

async function sendPrompt(rawPrompt) {
  const prompt = rawPrompt.trim();
  if (isSending) return;
  if (!prompt) {
    setComposerFeedback("Write a message before sending.", "error");
    promptInput.focus();
    return;
  }

  const scopeValue = Number.parseInt(scopeInput.value, 10);
  const payload = {
    prompt,
    retrieval_strategy: retrievalSelect.value,
    conversation: conversationHistory,
  };
  if (Number.isInteger(scopeValue) && scopeValue > 0) payload.scope_key = scopeValue;

  messages.querySelector(".welcome-message")?.remove();
  addUserMessage(prompt);
  promptInput.value = "";
  resizeComposer();
  setComposerFeedback();
  updateSendButton();
  setSending(true);
  const loadingMessage = addLoadingMessage();
  activeController = new AbortController();
  let finalResult = null;

  try {
    const response = await fetch("/api/v2/insights/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: activeController.signal,
    });
    if (!response.ok || !response.body) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || "The local API could not process that question.");
    }

    await consumeEventStream(response.body, (event, body) => {
      if (event === "status") updateLoadingMessage(loadingMessage, body);
      if (event === "result") finalResult = body;
      if (event === "error") throw new Error(body.detail || "The local API could not process that question.");
    });

    if (!finalResult) throw new Error("The local API ended before it returned an answer.");
    removeLoadingMessage(loadingMessage);
    addAssistantMessage(finalResult);
    rememberConversationTurn("user", prompt);
    rememberConversationTurn("assistant", finalResult.answer);
  } catch (error) {
    removeLoadingMessage(loadingMessage);
    const message =
      error instanceof DOMException && error.name === "AbortError"
        ? "Stopped the local request. You can ask a different question now."
        : error instanceof Error
          ? error.message
          : "Something went wrong while processing that question.";
    addAssistantMessage({}, message);
  } finally {
    activeController = null;
    setSending(false);
    promptInput.focus();
  }
}

function selectedScope() {
  return knownScopes.find((scope) => String(scope.scope_key) === scopeInput.value) || null;
}

function updateScopeSummary() {
  const scope = selectedScope();
  scopeSummary.textContent = scope
    ? `Searching ${scope.title} · ${scope.segment_count} indexed passages · Smart retrieval`
    : "Searching all documents · Smart retrieval";
  scopeList.querySelectorAll(".scope-card").forEach((card) => {
    card.classList.toggle("is-selected", card.dataset.scopeKey === scopeInput.value);
  });
}

function renderScopeList() {
  if (!knownScopes.length) {
    scopeList.innerHTML = `<p class="library-empty">No documents are indexed yet. Upload one to begin.</p>`;
    return;
  }

  scopeList.innerHTML = knownScopes
    .map(
      (scope) => `
        <div class="scope-row">
          <button class="scope-card ${String(scope.scope_key) === scopeInput.value ? "is-selected" : ""}" type="button" data-scope-key="${scope.scope_key}">
            <span class="scope-card-title">${escapeHtml(scope.title || "Untitled document")}</span>
            <span class="scope-card-meta">${scope.segment_count} passages${scope.is_uploaded ? " | uploaded" : ""}</span>
          </button>
          ${scope.is_uploaded ? `<button class="scope-delete" type="button" data-delete-scope-key="${scope.scope_key}" title="Delete uploaded document" aria-label="Delete ${escapeHtml(scope.title || "uploaded document")}">x</button>` : ""}
        </div>
      `,
    )
    .join("");

  scopeList.querySelectorAll(".scope-card").forEach((card) => {
    card.addEventListener("click", () => {
      scopeInput.value = card.dataset.scopeKey || "";
      updateScopeSummary();
      promptInput.focus();
    });
  });
  scopeList.querySelectorAll(".scope-delete").forEach((button) => {
    button.addEventListener("click", () => deleteUploadedScope(button.dataset.deleteScopeKey || ""));
  });
}

async function loadScopes(selectScopeKey = "") {
  scopeList.setAttribute("aria-busy", "true");
  try {
    const response = await fetch("/api/v2/knowledge/scopes");
    if (!response.ok) throw new Error("Unable to load documents");
    const body = await response.json();
    knownScopes = Array.isArray(body.scopes) ? body.scopes : [];
    const previousValue = selectScopeKey || scopeInput.value;
    scopeInput.innerHTML = `<option value="">All documents</option>${knownScopes
      .map((scope) => `<option value="${scope.scope_key}">${escapeHtml(scope.title || `Document ${scope.scope_key}`)}</option>`)
      .join("")}`;
    scopeInput.value = knownScopes.some((scope) => String(scope.scope_key) === String(previousValue))
      ? String(previousValue)
      : "";
    renderScopeList();
    updateScopeSummary();
  } catch {
    scopeList.innerHTML = `<p class="library-empty library-error">Could not load the local document library.</p>`;
  } finally {
    scopeList.setAttribute("aria-busy", "false");
  }
}

function setUploadStatus(message, state = "") {
  uploadStatus.textContent = message;
  uploadStatus.dataset.state = state;
}

function showSelectedFile() {
  const file = uploadFile.files?.[0];
  selectedFile.textContent = file ? `${file.name} (${formatBytes(file.size)})` : "No file selected";
}

function toggleUpload(forceOpen) {
  const shouldOpen = typeof forceOpen === "boolean" ? forceOpen : uploadForm.hidden;
  uploadForm.hidden = !shouldOpen;
  uploadToggle.setAttribute("aria-expanded", String(shouldOpen));
  if (shouldOpen) uploadTitle.focus();
}

async function uploadDocument() {
  const file = uploadFile.files?.[0];
  if (!file) {
    setUploadStatus("Choose a TXT, Markdown, or PDF file first.", "error");
    return;
  }

  const body = new FormData();
  body.append("file", file);
  if (uploadTitle.value.trim()) body.append("title", uploadTitle.value.trim());

  uploadButton.disabled = true;
  setUploadStatus("Reading the document locally...", "working");
  if (uploadProgressTimer) window.clearTimeout(uploadProgressTimer);
  uploadProgressTimer = window.setTimeout(
    () => setUploadStatus("Creating local embeddings for searchable passages...", "working"),
    900,
  );

  try {
    const response = await fetch("/api/v2/knowledge/scopes/file", { method: "POST", body });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || "Unable to index that document.");

    await loadScopes(String(result.scope_key));
    setUploadStatus(`Ready: ${result.segments_created} passages indexed in ${formatDuration(result.processing_time_ms)}.`, "success");
    showToast(`${result.segments_created} passages are ready to search.`);
    uploadForm.reset();
    showSelectedFile();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to index that document.";
    setUploadStatus(message, "error");
    showToast(message, "error");
  } finally {
    if (uploadProgressTimer) window.clearTimeout(uploadProgressTimer);
    uploadButton.disabled = false;
  }
}

async function deleteUploadedScope(scopeKey) {
  const scope = knownScopes.find((item) => String(item.scope_key) === String(scopeKey));
  if (!scope || !(await requestDeleteConfirmation(scope.title))) return;

  try {
    const response = await fetch(`/api/v2/knowledge/scopes/${encodeURIComponent(scopeKey)}`, {
      method: "DELETE",
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || "Unable to delete that document.");
    const selectedWasDeleted = scopeInput.value === String(scopeKey);
    await loadScopes(selectedWasDeleted ? "" : scopeInput.value);
    showToast(`Removed ${scope.title} and its indexed passages.`);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to delete that document.";
    setUploadStatus(message, "error");
    showToast(message, "error");
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (isSending && activeController) {
    activeController.abort();
    return;
  }
  sendPrompt(promptInput.value);
});

promptInput.addEventListener("input", () => {
  resizeComposer();
  setComposerFeedback();
  updateSendButton();
});
promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

scopeInput.addEventListener("change", updateScopeSummary);
uploadToggle.addEventListener("click", () => toggleUpload());
uploadForm.addEventListener("submit", (event) => {
  event.preventDefault();
  uploadDocument();
});
uploadFile.addEventListener("change", showSelectedFile);

commandTrigger.addEventListener("click", openCommandPalette);
commandClose.addEventListener("click", closeCommandPalette);
themeToggle.addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
});
commandInput.addEventListener("input", () => renderCommandResults(commandInput.value));
commandInput.addEventListener("keydown", (event) => {
  if (event.key === "ArrowDown") {
    event.preventDefault();
    setCommandSelection(selectedCommandIndex + 1);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    setCommandSelection(selectedCommandIndex - 1);
  } else if (event.key === "Enter") {
    event.preventDefault();
    activateCommand(commandItems[selectedCommandIndex]);
  }
});
commandResults.addEventListener("click", (event) => {
  const button = event.target.closest("[data-command-index]");
  if (!button) return;
  activateCommand(commandItems[Number(button.dataset.commandIndex)]);
});
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    if (commandPalette.open) closeCommandPalette();
    else openCommandPalette();
  }
  if (event.key === "/" && !commandPalette.open && document.activeElement === document.body) {
    event.preventDefault();
    promptInput.focus();
  }
});
commandPalette.addEventListener("click", (event) => {
  if (event.target === commandPalette) closeCommandPalette();
});
confirmDialog.addEventListener("click", (event) => {
  if (event.target === confirmDialog) confirmDialog.close("cancel");
});
sourceViewerClose.addEventListener("click", () => sourceViewer.close());
sourceViewer.addEventListener("click", (event) => {
  if (event.target === sourceViewer) sourceViewer.close();
});
sourceViewer.addEventListener("close", resetSourceViewer);
sourceViewerPrev.addEventListener("click", () => activeSource && showPdfPage(activeSource.page - 1));
sourceViewerNext.addEventListener("click", () => activeSource && showPdfPage(activeSource.page + 1));
sourceViewer.addEventListener("keydown", (event) => {
  if (activeSource?.kind !== "pdf" || event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.key === "ArrowLeft" && !sourceViewerPrev.disabled) {
    event.preventDefault();
    showPdfPage(activeSource.page - 1);
  } else if (event.key === "ArrowRight" && !sourceViewerNext.disabled) {
    event.preventDefault();
    showPdfPage(activeSource.page + 1);
  }
});
messages.addEventListener("click", (event) => {
  const openButton = event.target.closest(".citation-open");
  if (openButton) {
    openSource(
      openButton.dataset.sourceDocumentId,
      openButton.dataset.passageId || "",
      openButton.dataset.title || "",
    );
    return;
  }
  const button = event.target.closest(".onboarding-action");
  if (!button) return;
  if (button.dataset.command === "upload") {
    toggleUpload(true);
    return;
  }
  if (button.dataset.quickPrompt) {
    promptInput.value = button.dataset.quickPrompt;
    resizeComposer();
    updateSendButton();
    promptInput.focus();
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  });
});
dropzone.addEventListener("drop", (event) => {
  const files = event.dataTransfer?.files;
  if (!files?.length) return;
  uploadFile.files = files;
  showSelectedFile();
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    promptInput.value = button.dataset.prompt || "";
    resizeComposer();
    updateSendButton();
    promptInput.focus();
  });
});

newChatButton.addEventListener("click", () => {
  if (isSending && activeController) activeController.abort();
  conversationHistory = [];
  messages.replaceChildren();
  messages.insertAdjacentHTML("afterbegin", welcomeMarkup());
  promptInput.value = "";
  resizeComposer();
  setComposerFeedback();
  updateSendButton();
  promptInput.focus();
});

async function refreshHealth() {
  try {
    const response = await fetch("/health/detailed");
    if (!response.ok) throw new Error("Health request failed");
    const health = await response.json();
    const llm = health.components?.llm || {};
    const provider = llm.provider || "local model";
    const model = llm.chat_model || "model";
    providerChip.textContent = `${provider} | ${model}`;
    localStatus.innerHTML = `<span class="status-dot" aria-hidden="true"></span><span>${escapeHtml(provider)} connected locally</span>`;
  } catch {
    localStatus.classList.add("unavailable");
    localStatus.innerHTML = `<span class="status-dot" aria-hidden="true"></span><span>Local API unavailable</span>`;
  }
}

resizeComposer();
applyTheme(preferredTheme(), false);
updateSendButton();
refreshHealth();
loadScopes();
