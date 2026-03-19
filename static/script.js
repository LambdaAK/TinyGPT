// Client-side conversation state
let messages = [];
let isStreaming = false;

// DOM refs
const messagesEl = document.getElementById("messages");
const emptyState = document.getElementById("emptyState");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const settingsToggle = document.getElementById("settingsToggle");
const settingsBody = document.getElementById("settingsBody");
const chevron = document.getElementById("chevron");
const tempSlider = document.getElementById("temperature");
const topkSlider = document.getElementById("topk");
const tempVal = document.getElementById("tempVal");
const topkVal = document.getElementById("topkVal");
const modelInfo = document.getElementById("modelInfo");

// Load model info on page load
fetch("/info")
  .then(r => r.json())
  .then(data => {
    if (data.checkpoint) {
      modelInfo.innerHTML = `
        <span class="model-name">${data.checkpoint}</span>
        <span style="color:var(--text-dim);font-size:11px;display:block">
          epoch ${data.epoch} · loss ${data.val_loss} · ${(data.parameters / 1000).toFixed(0)}K params
        </span>`;
    }
  })
  .catch(() => {});

// Settings toggle
settingsToggle.addEventListener("click", () => {
  const open = settingsBody.classList.toggle("open");
  chevron.classList.toggle("open", open);
});

// Slider live update
tempSlider.addEventListener("input", () => { tempVal.textContent = tempSlider.value; });
topkSlider.addEventListener("input", () => { topkVal.textContent = topkSlider.value; });

// New chat
newChatBtn.addEventListener("click", () => {
  if (isStreaming) return;
  messages = [];
  messagesEl.innerHTML = "";
  messagesEl.appendChild(emptyState);
  emptyState.style.display = "";
});

// Auto-resize textarea
userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 200) + "px";
  sendBtn.disabled = !userInput.value.trim() || isStreaming;
});

// Send on Enter (Shift+Enter for newline)
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);

function sendExample(text) {
  userInput.value = text;
  userInput.style.height = "auto";
  sendBtn.disabled = false;
  sendMessage();
}

function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isStreaming) return;

  // Hide empty state
  emptyState.style.display = "none";

  // Add user message to state and UI
  messages.push({ role: "user", content: text });
  appendUserBubble(text);

  // Clear input
  userInput.value = "";
  userInput.style.height = "auto";
  sendBtn.disabled = true;

  // Stream assistant response
  streamResponse();
}

function appendUserBubble(text) {
  const row = document.createElement("div");
  row.className = "message-row user";
  const bubble = document.createElement("div");
  bubble.className = "user-bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
}

function appendAssistantRow() {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  const content = document.createElement("div");
  content.className = "assistant-content";
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  content.appendChild(cursor);
  row.appendChild(content);
  messagesEl.appendChild(row);
  scrollToBottom();
  return { content, cursor };
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function streamResponse() {
  isStreaming = true;
  const { content, cursor } = appendAssistantRow();
  let responseText = "";

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages,
        temperature: parseFloat(tempSlider.value),
        top_k: parseInt(topkSlider.value),
      }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      content.textContent = "[Error: " + (err.error || resp.statusText) + "]";
      content.style.color = "#e55";
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buf += decoder.decode(value, { stream: true });

      // Parse SSE lines
      const parts = buf.split("\n\n");
      buf = parts.pop(); // keep incomplete chunk

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data: ")) continue;
        const token = line.slice(6); // strip "data: "

        if (token === "[DONE]") break;

        responseText += token;

        // Update display: remove cursor, set text, re-add cursor
        cursor.remove();
        content.textContent = responseText;
        content.appendChild(cursor);
        scrollToBottom();
      }
    }
  } catch (err) {
    content.textContent = "[Connection error]";
    content.style.color = "#e55";
  } finally {
    // Remove blinking cursor, finalize
    cursor.remove();
    if (!responseText) {
      content.textContent = "...";
      content.style.color = "var(--text-muted)";
    }

    // Save to conversation history
    messages.push({ role: "assistant", content: responseText || "..." });

    isStreaming = false;
    sendBtn.disabled = !userInput.value.trim();
  }
}
