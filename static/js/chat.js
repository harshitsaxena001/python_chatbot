class ProfessionalChatInterface {
  constructor() {
    this.messageInput = document.getElementById("messageInput");
    this.sendButton = document.getElementById("sendButton");
    this.messagesContainer = document.getElementById("messagesContainer");
    this.typingIndicator = document.getElementById("typingIndicator");
    this.correctionModal = document.getElementById("correctionModal");
    this.statsModal = document.getElementById("statsModal");

    this.currentCorrectionData = null;

    this.initializeEventListeners();
    this.loadSessionStats();
  }

  initializeEventListeners() {
    // Send message events
    this.sendButton.addEventListener("click", () => this.sendMessage());

    // Enhanced keyboard handling
    this.messageInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Auto-resize textarea
    this.messageInput.addEventListener("input", () => {
      this.autoResizeTextarea();
    });

    // Modal close events
    document.addEventListener("click", (e) => {
      if (e.target.classList.contains("modal")) {
        this.closeModals();
      }
    });
  }

  autoResizeTextarea() {
    const textarea = this.messageInput;
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
  }

  async sendMessage() {
    const message = this.messageInput.value.trim();
    if (!message) return;

    // Disable input
    this.setInputState(false);

    // Add user message
    this.addMessage(message, "user");

    // Clear input
    this.messageInput.value = "";
    this.autoResizeTextarea();

    // Show typing indicator
    this.showTypingIndicator();

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message }),
      });

      const data = await response.json();

      if (response.ok) {
        this.hideTypingIndicator();

        // Add bot response
        const messageElement = this.addMessage(
          data.response,
          "bot",
          data.timestamp,
          data.session_corrected,
          data.learned_from_history
        );

        // Add feedback buttons for regular responses
        if (
          !["demo_mode", "session_corrected", "learned_from_history"].includes(
            data.intent
          )
        ) {
          this.addFeedbackButtons(messageElement, message, data.response);
        }
      } else {
        this.hideTypingIndicator();
        this.addMessage(
          "Sorry, I encountered an error. Please try again.",
          "bot"
        );
      }
    } catch (error) {
      console.error("Error:", error);
      this.hideTypingIndicator();
      this.addMessage("Connection error. Please check your internet.", "bot");
    }

    this.setInputState(true);
  }

  addMessage(
    text,
    sender,
    timestamp = null,
    isSessionCorrected = false,
    isLearned = false
  ) {
    const messageDiv = document.createElement("div");
    let className = `message ${sender}-message`;

    if (isSessionCorrected) className += " message-corrected";
    if (isLearned) className += " message-learned";

    messageDiv.className = className;

    const currentTime =
      timestamp ||
      new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });

    let badges = "";
    if (isSessionCorrected)
      badges += '<span class="message-badge session">Session Memory</span>';
    if (isLearned)
      badges += '<span class="message-badge learned">Learned</span>';

    messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-${sender === "user" ? "user" : "robot"}"></i>
            </div>
            <div class="message-content">
                <div class="message-bubble">${this.escapeHtml(text)}</div>
                <div class="message-meta">
                    <div class="message-time">
                        <i class="fas fa-clock"></i>
                        ${currentTime} ${badges}
                    </div>
                    <div class="message-actions">
                        <button class="action-btn" onclick="chatInterface.copyMessage(this)" title="Copy">
                            <i class="fas fa-copy"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;

    this.messagesContainer.appendChild(messageDiv);
    this.scrollToBottom();

    return messageDiv;
  }

  addFeedbackButtons(messageElement, userMessage, botResponse) {
    const feedbackDiv = document.createElement("div");
    feedbackDiv.className = "feedback-buttons";
    feedbackDiv.innerHTML = `
            <button class="feedback-btn positive" onclick="chatInterface.handlePositiveFeedback('${this.escapeForAttribute(
              userMessage
            )}', '${this.escapeForAttribute(botResponse)}')">
                <i class="fas fa-thumbs-up"></i> Helpful
            </button>
            <button class="feedback-btn negative" onclick="chatInterface.handleNegativeFeedback('${this.escapeForAttribute(
              userMessage
            )}', '${this.escapeForAttribute(botResponse)}')">
                <i class="fas fa-edit"></i> Teach Better Answer
            </button>
        `;

    messageElement.querySelector(".message-content").appendChild(feedbackDiv);
  }

  handlePositiveFeedback(userMessage, botResponse) {
    this.sendFeedback(userMessage, botResponse, "positive");
    this.showNotification("Thank you for the positive feedback!", "success");
  }

  handleNegativeFeedback(userMessage, botResponse) {
    this.currentCorrectionData = { userMessage, botResponse };
    this.openCorrectionModal(userMessage, botResponse);
  }

  openCorrectionModal(userMessage, botResponse) {
    document.getElementById("originalQuestion").textContent = userMessage;
    document.getElementById("wrongAnswer").textContent = botResponse;
    document.getElementById("correctAnswer").value = "";
    this.correctionModal.style.display = "block";
  }

  closeCorrectionModal() {
    this.correctionModal.style.display = "none";
    this.currentCorrectionData = null;
  }

  async submitCorrection() {
    const correctAnswer = document.getElementById("correctAnswer").value.trim();

    if (!correctAnswer) {
      this.showNotification("Please provide the correct answer.", "warning");
      return;
    }

    await this.sendFeedback(
      this.currentCorrectionData.userMessage,
      this.currentCorrectionData.botResponse,
      "negative",
      correctAnswer
    );

    this.closeCorrectionModal();
    this.loadSessionStats(); // Refresh stats
  }

  async sendFeedback(
    userMessage,
    botResponse,
    feedbackType,
    correctResponse = null
  ) {
    try {
      const response = await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_message: userMessage,
          bot_response: botResponse,
          feedback: feedbackType,
          correct_response: correctResponse,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        this.showNotification(data.message, "success");
        // Remove feedback buttons after submission
        document
          .querySelectorAll(".feedback-buttons")
          .forEach((el) => el.remove());
      } else {
        this.showNotification("Error processing feedback.", "error");
      }
    } catch (error) {
      console.error("Error:", error);
      this.showNotification("Connection error.", "error");
    }
  }

  async loadSessionStats() {
    try {
      const response = await fetch("/session_stats");
      const data = await response.json();

      document.getElementById("correctionsCount").textContent =
        data.session_corrections;
      document.getElementById("conversationsCount").textContent = Math.floor(
        data.session_conversations / 2
      );

      this.displayRecentCorrections(data.recent_corrections);
    } catch (error) {
      console.error("Error loading stats:", error);
    }
  }

  displayRecentCorrections(corrections) {
    const container = document.getElementById("recentCorrections");

    if (corrections.length === 0) {
      container.innerHTML =
        '<p class="text-muted">No corrections in this session yet.</p>';
      return;
    }

    let html = '<h4>Recent Corrections:</h4><div class="corrections-list">';

    corrections.forEach((correction) => {
      html += `
                <div class="correction-item">
                    <div class="correction-question">Q: ${
                      correction.user_message
                    }</div>
                    <div class="correction-answer">A: ${
                      correction.correct_response
                    }</div>
                    <div class="correction-time">${new Date(
                      correction.timestamp
                    ).toLocaleString()}</div>
                </div>
            `;
    });

    html += "</div>";
    container.innerHTML = html;
  }

  // Utility methods
  showStats() {
    this.loadSessionStats();
    this.statsModal.style.display = "block";
  }

  closeStatsModal() {
    this.statsModal.style.display = "none";
  }

  async clearSession() {
    if (confirm("Clear all session data including corrections?")) {
      try {
        const response = await fetch("/clear_session", { method: "POST" });
        const data = await response.json();

        if (response.ok) {
          location.reload();
        } else {
          this.showNotification("Error clearing session.", "error");
        }
      } catch (error) {
        console.error("Error:", error);
        this.showNotification("Connection error.", "error");
      }
    }
  }

  exportChat() {
    const messages = this.messagesContainer.querySelectorAll(".message");
    let chatHistory = "AI Assistant Pro - Chat Export\n";
    chatHistory += "=====================================\n\n";

    messages.forEach((message) => {
      const isUser = message.classList.contains("user-message");
      const sender = isUser ? "You" : "AI Assistant";
      const text = message.querySelector(".message-bubble").textContent.trim();
      const time = message.querySelector(".message-time").textContent.trim();

      chatHistory += `[${time}] ${sender}:\n${text}\n\n`;
    });

    const blob = new Blob([chatHistory], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chat-export-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.showNotification("Chat exported successfully!", "success");
  }

  copyMessage(button) {
    const text = button
      .closest(".message-content")
      .querySelector(".message-bubble").textContent;
    navigator.clipboard.writeText(text).then(() => {
      this.showNotification("Message copied!", "info");
    });
  }

  showNotification(message, type = "info") {
    // Create notification element
    const notification = document.createElement("div");
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
            <i class="fas fa-${
              type === "success"
                ? "check"
                : type === "error"
                ? "exclamation"
                : "info"
            }"></i>
            <span>${message}</span>
        `;

    // Add to body
    document.body.appendChild(notification);

    // Remove after 3 seconds
    setTimeout(() => {
      notification.remove();
    }, 3000);
  }

  showQuickActions() {
    const actions = [
      "What are the library hours?",
      "Calculate 15% of 500",
      "What's the square root of 144?",
      "Tell me about campus facilities",
    ];

    const randomAction = actions[Math.floor(Math.random() * actions.length)];
    this.messageInput.value = randomAction;
    this.messageInput.focus();
  }

  setInputState(enabled) {
    this.messageInput.disabled = !enabled;
    this.sendButton.disabled = !enabled;
    if (enabled) {
      this.messageInput.focus();
    }
  }

  showTypingIndicator() {
    this.typingIndicator.style.display = "flex";
    this.scrollToBottom();
  }

  hideTypingIndicator() {
    this.typingIndicator.style.display = "none";
  }

  scrollToBottom() {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }

  closeModals() {
    this.correctionModal.style.display = "none";
    this.statsModal.style.display = "none";
  }

  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  escapeForAttribute(text) {
    return text.replace(/'/g, "\\'").replace(/"/g, '\\"');
  }
}

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  window.chatInterface = new ProfessionalChatInterface();
});

// Global functions for onclick handlers
function showStats() {
  window.chatInterface.showStats();
}

function closeStatsModal() {
  window.chatInterface.closeStatsModal();
}

function closeCorrectionModal() {
  window.chatInterface.closeCorrectionModal();
}

function submitCorrection() {
  window.chatInterface.submitCorrection();
}

function clearSession() {
  window.chatInterface.clearSession();
}

function exportChat() {
  window.chatInterface.exportChat();
}

function showQuickActions() {
  window.chatInterface.showQuickActions();
}
