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

let isSending = false;
let activeController = null;
let knownScopes = [];
let uploadProgressTimer = null;
let conversationHistory = [];

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

function updateSendButton() {
  if (isSending) {
    sendButton.disabled = false;
    return;
  }
  sendButton.disabled = !promptInput.value.trim();
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
        <p class="message-label">PolicyMind</p>
        <div class="message-body">
          <p>New conversation started. Chat normally, or ask about a policy, your uploaded document, or service-case data.</p>
          <p class="muted">I only search local sources when the question actually needs them.</p>
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
  messages.append(fragment);
  scrollToLatest();
  return element;
}

function removeLoadingMessage(element) {
  if (!element) return;
  if (element._elapsedTimer) window.clearInterval(element._elapsedTimer);
  element.remove();
}

function resultMeta(result) {
  const type = String(result.query_type || "document_qa").replaceAll("_", " ");
  const latency = Number.isFinite(result.latency_ms)
    ? `<span class="latency-chip">${formatDuration(result.latency_ms)} response time</span>`
    : "";
  return `<span class="query-type">${escapeHtml(type)}</span>${latency}`;
}

function citationsMarkup(citations) {
  if (!Array.isArray(citations) || citations.length === 0) return "";

  const items = citations
    .map(
      (citation, index) => `
        <details class="citation">
          <summary>
            <span>Source ${index + 1}</span>
            <span>${escapeHtml(citation.contract_title || "Policy document")}</span>
          </summary>
          <p class="citation-location">${escapeHtml(citation.section || "Indexed policy passage")}${
            citation.page ? ` | page ${escapeHtml(citation.page)}` : ""
          }</p>
          <p>${escapeHtml(citation.chunk_text || "")}</p>
        </details>
      `,
    )
    .join("");

  return `<section class="sources"><p class="source-heading">Evidence used</p>${items}</section>`;
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
  sendIcon.textContent = sending ? "x" : "↑";
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
    ? `Focused on ${scope.title} (${scope.segment_count} indexed passages).`
    : "Normal conversation stays in chat; policy and data questions use your local sources.";
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
    uploadForm.reset();
    showSelectedFile();
  } catch (error) {
    setUploadStatus(error instanceof Error ? error.message : "Unable to index that document.", "error");
  } finally {
    if (uploadProgressTimer) window.clearTimeout(uploadProgressTimer);
    uploadButton.disabled = false;
  }
}

async function deleteUploadedScope(scopeKey) {
  const scope = knownScopes.find((item) => String(item.scope_key) === String(scopeKey));
  if (!scope || !window.confirm(`Delete "${scope.title}" and its indexed passages?`)) return;

  try {
    const response = await fetch(`/api/v2/knowledge/scopes/${encodeURIComponent(scopeKey)}`, {
      method: "DELETE",
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || "Unable to delete that document.");
    const selectedWasDeleted = scopeInput.value === String(scopeKey);
    await loadScopes(selectedWasDeleted ? "" : scopeInput.value);
  } catch (error) {
    setUploadStatus(error instanceof Error ? error.message : "Unable to delete that document.", "error");
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
updateSendButton();
refreshHealth();
loadScopes();
