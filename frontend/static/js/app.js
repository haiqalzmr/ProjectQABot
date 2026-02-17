/**
 * Policy Q&A Bot — Chat Interface
 * Chat interface: collapsible sidebar, centered welcome, user bubbles,
 * localStorage chat history + theme persistence.
 */

(function () {
    "use strict";

    // ── SVG Icons ─────────────────────────────────────────────────────────
    var USER_ICON_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>';

    // Robot icon for AI avatar (matches the reference picture)
    var AI_ICON_SVG = '<svg viewBox="0 0 32 32" fill="none">' +
        '<rect x="4" y="10" width="24" height="16" rx="4" stroke="currentColor" stroke-width="2.5" fill="none"/>' +
        '<circle cx="12" cy="18" r="2.5" fill="currentColor"/>' +
        '<circle cx="20" cy="18" r="2.5" fill="currentColor"/>' +
        '<line x1="16" y1="4" x2="16" y2="10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>' +
        '<circle cx="16" cy="3" r="2" fill="currentColor"/>' +
        '<line x1="1" y1="16" x2="4" y2="16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>' +
        '<line x1="28" y1="16" x2="31" y2="16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>' +
        '</svg>';

    var LIGHTBULB_SVG = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18h6"></path><path d="M10 22h4"></path><path d="M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z"></path></svg>';
    var CHAT_ICON_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>';

    // ── DOM References ────────────────────────────────────────────────────
    var chatContainer = document.getElementById("chatContainer");
    var messagesDiv = document.getElementById("messages");
    var welcomeScreen = document.getElementById("welcomeScreen");
    var queryInput = document.getElementById("queryInput");
    var sendBtn = document.getElementById("sendBtn");
    var welcomeQueryInput = document.getElementById("welcomeQueryInput");
    var welcomeSendBtn = document.getElementById("welcomeSendBtn");
    var inputArea = document.getElementById("inputArea");
    var newChatBtn = document.getElementById("newChatBtn");
    var sidebar = document.getElementById("sidebar");
    var sidebarCollapseBtn = document.getElementById("sidebarCollapseBtn");
    var sidebarExpandBtn = document.getElementById("sidebarExpandBtn");
    var themeToggle = document.getElementById("themeToggle");
    var chatHistoryList = document.getElementById("chatHistoryList");

    // ── State ─────────────────────────────────────────────────────────────
    var isLoading = false;
    var currentChatId = null;
    var currentMessages = [];
    var STORAGE_KEY = "policybot_chats";
    var THEME_KEY = "policybot_theme";
    var SIDEBAR_KEY = "policybot_sidebar";


    // ══════════════════════════════════════════════════════════════════════
    // SIDEBAR COLLAPSE / EXPAND
    // ══════════════════════════════════════════════════════════════════════
    function initSidebar() {
        var saved = localStorage.getItem(SIDEBAR_KEY);
        if (saved === "collapsed") {
            collapseSidebar();
        }
    }

    function collapseSidebar() {
        sidebar.classList.remove("open");
        sidebarExpandBtn.classList.add("visible");
        localStorage.setItem(SIDEBAR_KEY, "collapsed");
    }

    function expandSidebar() {
        sidebar.classList.add("open");
        sidebarExpandBtn.classList.remove("visible");
        localStorage.setItem(SIDEBAR_KEY, "expanded");
    }

    sidebarCollapseBtn.addEventListener("click", collapseSidebar);
    sidebarExpandBtn.addEventListener("click", expandSidebar);


    // ══════════════════════════════════════════════════════════════════════
    // THEME MANAGEMENT
    // ══════════════════════════════════════════════════════════════════════
    function initTheme() {
        var saved = localStorage.getItem(THEME_KEY);
        document.documentElement.setAttribute("data-theme", saved || "light");
    }

    function toggleTheme() {
        var current = document.documentElement.getAttribute("data-theme");
        var next = current === "light" ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", next);
        localStorage.setItem(THEME_KEY, next);
    }

    themeToggle.addEventListener("click", toggleTheme);


    // ══════════════════════════════════════════════════════════════════════
    // CHAT HISTORY (localStorage)
    // ══════════════════════════════════════════════════════════════════════
    function getAllChats() {
        try {
            var data = localStorage.getItem(STORAGE_KEY);
            return data ? JSON.parse(data) : [];
        } catch (e) { return []; }
    }

    function saveAllChats(chats) {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
        } catch (e) {
            if (chats.length > 1) {
                chats.shift();
                saveAllChats(chats);
            }
        }
    }

    function generateChatId() {
        return "chat_" + Date.now() + "_" + Math.random().toString(36).substr(2, 6);
    }

    function getChatTitle(messages) {
        for (var i = 0; i < messages.length; i++) {
            if (messages[i].role === "user") {
                var title = messages[i].content;
                return title.length > 36 ? title.substring(0, 33) + "..." : title;
            }
        }
        return "New Chat";
    }

    function saveCurrentChat(messages) {
        if (!messages || messages.length === 0) return;
        var chats = getAllChats();
        var existingIndex = -1;

        for (var i = 0; i < chats.length; i++) {
            if (chats[i].id === currentChatId) { existingIndex = i; break; }
        }

        var chatData = {
            id: currentChatId,
            title: getChatTitle(messages),
            messages: messages,
            updatedAt: new Date().toISOString(),
        };

        if (existingIndex >= 0) {
            chats[existingIndex] = chatData;
        } else {
            chats.push(chatData);
        }

        if (chats.length > 20) chats = chats.slice(chats.length - 20);
        saveAllChats(chats);
        renderChatHistory();
    }

    function loadChat(chatId) {
        // Save current chat before switching
        if (currentMessages.length > 0 && currentChatId) {
            saveCurrentChat(currentMessages);
        }
        var chats = getAllChats();
        for (var i = 0; i < chats.length; i++) {
            if (chats[i].id === chatId) {
                currentChatId = chatId;
                currentMessages = chats[i].messages.slice(); // copy
                displaySavedChat(chats[i].messages);
                renderChatHistory();
                return;
            }
        }
    }

    /**
     * DELETE a chat from history.
     * Key fix: clear currentMessages/currentChatId BEFORE calling resetToWelcome
     * so that startNewChat doesn't re-save the deleted chat.
     */
    function deleteChat(chatId) {
        var chats = getAllChats();
        var filtered = [];
        for (var i = 0; i < chats.length; i++) {
            if (chats[i].id !== chatId) filtered.push(chats[i]);
        }
        saveAllChats(filtered);

        // If we just deleted the currently active chat, reset view
        if (currentChatId === chatId) {
            currentChatId = null;
            currentMessages = [];
            resetToWelcome();
        }
        renderChatHistory();
    }

    /** Reset the view to welcome screen without saving anything. */
    function resetToWelcome() {
        messagesDiv.innerHTML = "";
        messagesDiv.style.display = "none";
        welcomeScreen.style.display = "flex";
        inputArea.style.display = "none";
        queryInput.value = "";
        welcomeQueryInput.value = "";
        queryInput.style.height = "auto";
        welcomeQueryInput.focus();
    }

    function displaySavedChat(messages) {
        messagesDiv.innerHTML = "";
        welcomeScreen.style.display = "none";
        messagesDiv.style.display = "block";
        inputArea.style.display = "block";

        for (var i = 0; i < messages.length; i++) {
            var msg = messages[i];
            addMessageToDOM(msg.role, msg.content, msg.sources || [], msg.followUps || [], false);
        }
        scrollToBottom();
    }

    function renderChatHistory() {
        var chats = getAllChats();
        chatHistoryList.innerHTML = "";

        if (chats.length === 0) {
            var empty = document.createElement("li");
            empty.className = "chat-history-empty";
            empty.textContent = "No conversations yet";
            chatHistoryList.appendChild(empty);
            return;
        }

        for (var i = chats.length - 1; i >= 0; i--) {
            (function (chat) {
                var li = document.createElement("li");
                li.className = "chat-history-item";
                if (chat.id === currentChatId) li.className += " active";

                var icon = document.createElement("span");
                icon.innerHTML = CHAT_ICON_SVG;
                icon.style.flexShrink = "0";
                icon.style.display = "flex";

                var title = document.createElement("span");
                title.textContent = chat.title;

                var deleteBtn = document.createElement("span");
                deleteBtn.className = "chat-delete";
                deleteBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';
                deleteBtn.addEventListener("click", function (e) {
                    e.stopPropagation();
                    e.preventDefault();
                    deleteChat(chat.id);
                });

                li.appendChild(icon);
                li.appendChild(title);
                li.appendChild(deleteBtn);
                li.addEventListener("click", function () { loadChat(chat.id); });
                chatHistoryList.appendChild(li);
            })(chats[i]);
        }
    }


    // ══════════════════════════════════════════════════════════════════════
    // MARKDOWN RENDERER
    // ══════════════════════════════════════════════════════════════════════
    function renderMarkdown(text) {
        var html = text;
        html = html.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)/g, "<em>$1</em>");
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
        html = html.replace(/^### (.+)$/gm, '<h4 class="md-h4">$1</h4>');
        html = html.replace(/^## (.+)$/gm, '<h3 class="md-h3">$1</h3>');
        html = html.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");
        html = html.replace(/<\/blockquote>\s*<blockquote>/g, "<br>");
        html = html.replace(/^- (.+)$/gm, "<li>$1</li>");
        html = html.replace(/(<li>.*?<\/li>(\s*<li>.*?<\/li>)*)/gs, "<ul>$1</ul>");
        html = html.replace(/<\/ul>\s*<ul>/g, "");

        html = html.split(/\n\n+/).map(function (block) {
            block = block.trim();
            if (!block) return "";
            if (/^<[hublo]/i.test(block)) return block;
            return "<p>" + block.replace(/\n/g, "<br>") + "</p>";
        }).join("");

        return html;
    }


    // ══════════════════════════════════════════════════════════════════════
    // MESSAGE RENDERING
    // ══════════════════════════════════════════════════════════════════════
    function addMessageToDOM(role, content, sources, followUps, animate) {
        welcomeScreen.style.display = "none";
        messagesDiv.style.display = "block";
        inputArea.style.display = "block";

        var messageDiv = document.createElement("div");
        messageDiv.className = "message " + role;
        if (!animate) messageDiv.style.animation = "none";

        // Avatar
        var avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.innerHTML = role === "user" ? USER_ICON_SVG : AI_ICON_SVG;

        // Content
        var contentDiv = document.createElement("div");
        contentDiv.className = "message-content";

        if (role === "user") {
            contentDiv.innerHTML = escapeHtml(content);
        } else {
            var isNoAnswer = content.indexOf("couldn't find") !== -1 ||
                content.indexOf("I cannot find") !== -1;

            if (isNoAnswer) {
                contentDiv.innerHTML = renderNoAnswer(content);
            } else {
                var parts = content.split(/\nSources:\s*/);
                contentDiv.innerHTML = renderMarkdown(parts[0]);
            }

            if (sources && sources.length > 0) {
                contentDiv.appendChild(createCitationBlock(sources));
            }
            if (followUps && followUps.length > 0) {
                contentDiv.appendChild(createFollowUpBlock(followUps));
            }
        }

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        messagesDiv.appendChild(messageDiv);
        if (animate) scrollToBottom();
    }

    function addMessage(role, content, sources, followUps) {
        addMessageToDOM(role, content, sources || [], followUps || [], true);
    }

    function renderNoAnswer(text) {
        var parts = text.split(/\nSources:\s*/);
        return '<div class="no-answer">' +
            '<div class="no-answer-title">' + LIGHTBULB_SVG + ' Hmm, let me help</div>' +
            renderMarkdown(parts[0]) +
            '</div>';
    }

    function createCitationBlock(sources) {
        var block = document.createElement("div");
        block.className = "citation-block";

        var header = document.createElement("div");
        header.className = "citation-header";
        header.innerHTML =
            '<span class="citation-header-left">' +
            '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
            '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>' +
            '<polyline points="14 2 14 8 20 8"></polyline>' +
            '</svg> ' +
            sources.length + ' Source' + (sources.length > 1 ? 's' : '') + ' Referenced' +
            '</span>' +
            '<span class="citation-toggle">▼</span>';

        var body = document.createElement("div");
        body.className = "citation-body";

        sources.forEach(function (src) {
            var card = document.createElement("div");
            card.className = "source-card";

            var meta = [];
            if (src.clause) meta.push("§" + src.clause);
            if (src.section) meta.push(src.section);
            meta.push("Page " + src.page);
            if (src.score) meta.push("Score: " + (src.score * 100).toFixed(0) + "%");

            var snippetHtml = "";
            if (src.snippet) {
                var snip = src.snippet.substring(0, 200);
                if (src.snippet.length > 200) snip += "...";
                snippetHtml = '<div class="source-snippet">"' + escapeHtml(snip) + '"</div>';
            }

            card.innerHTML =
                '<div class="source-doc">' + escapeHtml(src.doc_name) + '</div>' +
                '<div class="source-meta">' + meta.join(" · ") + '</div>' + snippetHtml;
            body.appendChild(card);
        });

        header.addEventListener("click", function () {
            body.classList.toggle("open");
            header.querySelector(".citation-toggle").classList.toggle("open");
        });

        block.appendChild(header);
        block.appendChild(body);
        return block;
    }

    function createFollowUpBlock(followUps) {
        var block = document.createElement("div");
        block.className = "follow-up-block";

        var label = document.createElement("div");
        label.className = "follow-up-label";
        label.textContent = "Related questions";
        block.appendChild(label);

        var btnContainer = document.createElement("div");
        btnContainer.className = "follow-up-buttons";

        followUps.forEach(function (q) {
            var btn = document.createElement("button");
            btn.className = "follow-up-btn";
            btn.textContent = q;
            btn.addEventListener("click", function () {
                askQuestion(q);
            });
            btnContainer.appendChild(btn);
        });

        block.appendChild(btnContainer);
        return block;
    }

    function addTypingIndicator() {
        var div = document.createElement("div");
        div.className = "message assistant";
        div.id = "typingIndicator";
        div.innerHTML =
            '<div class="message-avatar">' + AI_ICON_SVG + '</div>' +
            '<div class="message-content">' +
            '<div class="typing-indicator">' +
            '<span class="typing-dot"></span>' +
            '<span class="typing-dot"></span>' +
            '<span class="typing-dot"></span>' +
            '</div>' +
            '</div>';
        messagesDiv.appendChild(div);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        var el = document.getElementById("typingIndicator");
        if (el) el.remove();
    }


    // ══════════════════════════════════════════════════════════════════════
    // API CALL
    // ══════════════════════════════════════════════════════════════════════
    function askQuestion(question) {
        if (isLoading || !question.trim()) return;
        isLoading = true;
        sendBtn.disabled = true;
        welcomeSendBtn.disabled = true;

        if (!currentChatId) currentChatId = generateChatId();

        addMessage("user", question);
        currentMessages.push({ role: "user", content: question });
        addTypingIndicator();

        fetch("/api/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question }),
        })
            .then(function (response) {
                removeTypingIndicator();
                if (!response.ok) {
                    return response.json().then(function (err) {
                        addMessage("assistant", "**Error:** " + (err.error || "Something went wrong."), [], []);
                        throw new Error("API error");
                    });
                }
                return response.json();
            })
            .then(function (data) {
                var sources = data.sources || [];
                var followUps = data.follow_ups || [];
                addMessage("assistant", data.answer, sources, followUps);
                currentMessages.push({
                    role: "assistant", content: data.answer,
                    sources: sources, followUps: followUps,
                });
                saveCurrentChat(currentMessages);
            })
            .catch(function (err) {
                if (err.message !== "API error") {
                    removeTypingIndicator();
                    addMessage("assistant", "**Error:** Could not connect to the server.", [], []);
                }
            })
            .finally(function () {
                isLoading = false;
                sendBtn.disabled = false;
                welcomeSendBtn.disabled = false;
                queryInput.focus();
            });
    }


    // ══════════════════════════════════════════════════════════════════════
    // UTILITIES
    // ══════════════════════════════════════════════════════════════════════
    function escapeHtml(text) {
        var div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function scrollToBottom() {
        requestAnimationFrame(function () {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        });
    }

    function autoResize(textarea) {
        textarea.style.height = "auto";
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
    }

    function startNewChat() {
        // Save current chat if it has messages
        if (currentMessages.length > 0 && currentChatId) {
            saveCurrentChat(currentMessages);
        }
        currentMessages = [];
        currentChatId = null;
        resetToWelcome();
        renderChatHistory();
    }


    // ══════════════════════════════════════════════════════════════════════
    // EVENT LISTENERS
    // ══════════════════════════════════════════════════════════════════════

    // Bottom input (during chat)
    sendBtn.addEventListener("click", function () {
        var q = queryInput.value.trim();
        if (q) { queryInput.value = ""; queryInput.style.height = "auto"; askQuestion(q); }
    });

    queryInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendBtn.click(); }
    });

    queryInput.addEventListener("input", function () { autoResize(queryInput); });

    // Welcome input
    welcomeSendBtn.addEventListener("click", function () {
        var q = welcomeQueryInput.value.trim();
        if (q) { welcomeQueryInput.value = ""; askQuestion(q); }
    });

    welcomeQueryInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); welcomeSendBtn.click(); }
    });

    welcomeQueryInput.addEventListener("input", function () { autoResize(welcomeQueryInput); });

    // New chat
    newChatBtn.addEventListener("click", startNewChat);

    // Suggestion cards (replaces old example-pill)
    var suggestionCards = document.querySelectorAll(".suggestion-card");
    suggestionCards.forEach(function (card) {
        card.addEventListener("click", function () {
            var question = card.getAttribute("data-question");
            askQuestion(question);
        });
    });

    // Close sidebar on outside click (mobile)
    document.addEventListener("click", function (e) {
        if (window.innerWidth <= 768 &&
            sidebar.classList.contains("open") &&
            !sidebar.contains(e.target) &&
            e.target !== sidebarExpandBtn) {
            collapseSidebar();
        }
    });


    // ══════════════════════════════════════════════════════════════════════
    // INITIALIZATION
    // ══════════════════════════════════════════════════════════════════════
    initTheme();
    initSidebar();
    renderChatHistory();
    welcomeQueryInput.focus();

})();
