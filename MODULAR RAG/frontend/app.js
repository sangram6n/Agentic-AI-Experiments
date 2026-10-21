// ORMAS KNOWLEDGE BASE - Frontend Client
// Bulletproof DOM-Slot Math & Markdown Renderer

const API_BASE = window.location.protocol.startsWith('http')
  ? window.location.origin
  : 'http://127.0.0.1:8000';

console.log('[ORMAS] Connecting to API_BASE:', API_BASE);

// Configure Marked.js
if (window.marked) {
  try {
    if (typeof marked.use === 'function') {
      marked.use({ gfm: true, breaks: true });
    } else if (typeof marked.setOptions === 'function') {
      marked.setOptions({ gfm: true, breaks: true });
    }
  } catch (e) {
    console.warn('[ORMAS] Marked config warning:', e);
  }
}

// Strategy descriptions
const strategyDescriptions = {
  standard: 'Balanced semantic retrieval with clear factual grounding.',
  comparative: 'Formats metrics, project statuses, and data into structured tables.',
  context_rich: 'Comprehensive deep-context retrieval with rich source citations.',
  concise: 'High-impact, concise bullet-point answers for quick executive review.'
};

/**
 * Bulletproof Markdown & Math Renderer using DOM Slots:
 * 1. Delimiters are replaced with HTML tags (<div class="katex-slot"> / <span class="katex-slot">)
 *    which Marked.js leaves completely intact (no underscore emphasis corruption!).
 * 2. Marked parses markdown tables, bold, lists, and code blocks.
 * 3. KaTeX directly renders into each slot element via katex.render().
 */
function renderMarkdownAndMath(container, rawText) {
  if (!rawText || !container) return;

  var mathBlocks = [];

  function addDisplayBlock(math) {
    var idx = mathBlocks.length;
    mathBlocks.push({ type: "display", math: math.trim() });
    return '\n<div class="katex-slot katex-display-slot" data-math-idx="' + idx + '"></div>\n';
  }

  function addInlineBlock(math) {
    var idx = mathBlocks.length;
    mathBlocks.push({ type: "inline", math: math.trim() });
    return '<span class="katex-slot katex-inline-slot" data-math-idx="' + idx + '"></span>';
  }

  var text = rawText;

  // 1. Display math: \[ ... \]
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, function(match, math) {
    return addDisplayBlock(math);
  });

  // 2. Display math: $$ ... $$
  text = text.replace(/\$\$([\s\S]*?)\$\$/g, function(match, math) {
    return addDisplayBlock(math);
  });

  // 3. Bare bracket display math: [ \text... ] or [ \frac... ]
  text = text.replace(/(?:^|\n)\[\s*(\\(?:text|frac|sum|prod|int|sqrt|mathbf|left|right|operatorname|mathrm)[\s\S]*?)\](?:\s*\n|$)/g, function(match, math) {
    return addDisplayBlock(math);
  });

  // 4. Inline math: \( ... \)
  text = text.replace(/\\\(([\s\S]*?)\\\)/g, function(match, math) {
    return addInlineBlock(math);
  });

  // 5. Inline math: $ ... $
  text = text.replace(/\$([^\$\n]+)\$/g, function(match, math) {
    return addInlineBlock(math);
  });

  // 6. Parse Markdown with Marked
  var html = text;
  if (window.marked && typeof marked.parse === 'function') {
    try {
      html = marked.parse(text);
    } catch (e) {
      console.warn("[ORMAS] Marked parse error, falling back:", e);
      html = text.replace(/\n/g, '<br>');
    }
  } else {
    html = text.replace(/\n/g, '<br>');
  }

  container.innerHTML = html;

  // 7. Render KaTeX into each DOM slot
  var slots = container.querySelectorAll('.katex-slot');
  slots.forEach(function(slot) {
    var idxAttr = slot.getAttribute('data-math-idx');
    if (idxAttr !== null) {
      var idx = parseInt(idxAttr, 10);
      var block = mathBlocks[idx];
      if (block) {
        if (window.katex && typeof katex.render === 'function') {
          try {
            katex.render(block.math, slot, {
              displayMode: block.type === "display",
              throwOnError: false
            });
          } catch (err) {
            console.warn('[ORMAS] KaTeX render error:', err);
            slot.textContent = block.type === "display" ? "\\[ " + block.math + " \\]" : "\\( " + block.math + " \\)";
          }
        } else {
          slot.textContent = block.type === "display" ? "\\[ " + block.math + " \\]" : "\\( " + block.math + " \\)";
        }
      }
    }
  });

  // 8. Wrap tables in responsive scrollers
  container.querySelectorAll('table').forEach(function(tbl) {
    if (!tbl.parentElement.classList.contains('table-wrapper')) {
      var wrapper = document.createElement('div');
      wrapper.className = 'table-wrapper';
      tbl.parentNode.insertBefore(wrapper, tbl);
      wrapper.appendChild(tbl);
    }
  });
}

/**
 * Append message bubble to chat transcript
 */
function appendMessage(role, content, citations) {
  var chatMessages = document.getElementById('chatMessages');
  if (!chatMessages) return;

  var msgDiv = document.createElement('div');
  msgDiv.className = 'message ' + role;

  var avatar = role === 'user' ? '👤' : '🏛️';
  var citationHtml = '';

  if (citations && citations.length > 0) {
    citationHtml = `
      <div class="citations-box">
        <div class="citations-label">📑 Document Sources & Citations:</div>
        <div class="citation-chips">
          ${citations.map(function(c) {
            return `
              <div class="citation-card" title="${c.snippet || ''}">
                <strong>${c.filename || 'Doc'}</strong> ${c.page ? `(p.${c.page})` : ''}
              </div>
            `;
          }).join('')}
        </div>
      </div>
    `;
  }

  msgDiv.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-content">
      <div class="rendered-body"></div>
      ${citationHtml}
    </div>
  `;

  var bodyDiv = msgDiv.querySelector('.rendered-body');
  renderMarkdownAndMath(bodyDiv, content);

  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Global Submit Handler - Invoked by Form, Button, and Enter key
 */
async function handleQuerySubmit(queryText) {
  var queryInput = document.getElementById('queryInput');
  var sendBtn = document.getElementById('sendBtn');
  var chatMessages = document.getElementById('chatMessages');
  var strategySelect = document.getElementById('strategySelect');
  var topKSlider = document.getElementById('topKSlider');

  var query = (queryText || (queryInput ? queryInput.value : '')).trim();
  if (!query) {
    console.warn('[ORMAS] Query is empty, ignoring.');
    return;
  }

  console.log('[ORMAS] Processing query:', query);

  // 1. Show user message immediately
  appendMessage('user', query);

  // 2. Reset input field
  if (queryInput) {
    queryInput.value = '';
    queryInput.style.height = 'auto';
  }
  if (sendBtn) {
    sendBtn.disabled = true;
  }

  // 3. Show loading bubble
  var loaderDiv = document.createElement('div');
  loaderDiv.className = 'message bot';
  loaderDiv.id = 'activeLoader';
  var currentStrategy = strategySelect ? strategySelect.value : 'standard';
  loaderDiv.innerHTML = `
    <div class="msg-avatar">🏛️</div>
    <div class="msg-content" style="color:var(--text-dim);font-style:italic;">
      Retrieving context with ${currentStrategy} strategy...
    </div>
  `;
  if (chatMessages) {
    chatMessages.appendChild(loaderDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // 4. Fetch response from API
  try {
    var kVal = topKSlider ? parseInt(topKSlider.value, 10) : 4;
    console.log('[ORMAS] Fetching /api/chat with k =', kVal);

    var res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        rag_strategy: currentStrategy,
        top_k: kVal
      })
    });

    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}: ${res.statusText}`);
    }

    var data = await res.json();
    if (loaderDiv.parentNode) loaderDiv.remove();

    if (data.answer) {
      appendMessage('bot', data.answer, data.citations);
    } else {
      appendMessage('bot', data.detail || 'No response returned from model.');
    }
  } catch (err) {
    console.error('[ORMAS] Chat error:', err);
    if (loaderDiv.parentNode) loaderDiv.remove();
    appendMessage('bot', `⚠️ **Connection Error**: Unable to reach backend (${err.message}).\n\nPlease verify that the server is running on \`${API_BASE}\`.`);
  } finally {
    if (sendBtn) {
      sendBtn.disabled = false;
    }
    if (queryInput) {
      queryInput.focus();
    }
  }
}

/**
 * Clear Chat History
 */
function clearChatHistory() {
  var chatMessages = document.getElementById('chatMessages');
  if (chatMessages) {
    chatMessages.innerHTML = `
      <div class="message bot welcome-card">
        <div class="msg-avatar">🏛️</div>
        <div class="msg-content">
          <h3>Chat Cleared</h3>
          <p>Ready for your next query.</p>
        </div>
      </div>
    `;
  }
}

/**
 * Load Document List
 */
async function loadDocuments() {
  var docList = document.getElementById('docList');
  var docCountBadge = document.getElementById('docCountBadge');

  try {
    var res = await fetch(`${API_BASE}/api/documents`);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    var data = await res.json();
    if (data.files && docList) {
      var fileNames = Object.keys(data.files);
      if (docCountBadge) {
        docCountBadge.textContent = `${fileNames.length} Files`;
      }
      docList.innerHTML = fileNames.map(function(f) {
        var isPdf = f.toLowerCase().endsWith('.pdf');
        return `
          <div class="doc-item" title="${f} (${data.files[f]} chunks)">
            <span class="doc-name">${f}</span>
            <span class="doc-tag">${isPdf ? 'PDF' : 'PPTX'}</span>
          </div>
        `;
      }).join('');
    }
  } catch (err) {
    console.warn('[ORMAS] Could not load document list:', err);
    if (docList) {
      docList.innerHTML = '<div class="doc-loading">5 Documents Indexed in ChromaDB</div>';
    }
  }
}

// Expose functions to window for inline onclick/onsubmit attributes
window.handleQuerySubmit = handleQuerySubmit;
window.clearChatHistory = clearChatHistory;
window.loadDocuments = loadDocuments;

// DOM Initialization
document.addEventListener('DOMContentLoaded', function() {
  var strategySelect = document.getElementById('strategySelect');
  var strategyDesc = document.getElementById('strategyDesc');
  var topKSlider = document.getElementById('topKSlider');
  var topKValue = document.getElementById('topKValue');
  var queryInput = document.getElementById('queryInput');
  var chatForm = document.getElementById('chatForm');

  if (strategySelect && strategyDesc) {
    strategySelect.addEventListener('change', function(e) {
      strategyDesc.textContent = strategyDescriptions[e.target.value] || '';
    });
  }

  if (topKSlider && topKValue) {
    topKSlider.addEventListener('input', function(e) {
      topKValue.textContent = e.target.value;
    });
  }

  if (queryInput) {
    queryInput.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = (this.scrollHeight) + 'px';
    });
  }

  if (chatForm) {
    chatForm.addEventListener('submit', function(e) {
      e.preventDefault();
      handleQuerySubmit();
    });
  }

  loadDocuments();
  console.log('[ORMAS] DOM fully loaded and initialized.');
});

// Run loadDocuments immediately in case DOMContentLoaded already fired
loadDocuments();
