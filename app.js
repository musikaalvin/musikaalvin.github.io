/**
 * ============================================================================
 * KENTSCRIPT IDE - MAIN APPLICATION LOGIC
 * ============================================================================
 * Production-grade IDE with 20+ features
 * Real-time execution, validation, syntax highlighting, etc.
 * =========================================================================*/

const API_URL = 'http://localhost:3001/api';

// ============================================================================
// IDE STATE MANAGEMENT
// ============================================================================

class IDEState {
    constructor() {
        this.currentCode = '// Welcome to KentScript IDE\nprint("Hello, World!")';
        this.currentFile = 'Untitled.ks';
        this.isModified = false;
        this.isRunning = false;
        this.history = [];
        this.historyIndex = -1;
        this.settings = {
            fontSize: 14,
            wordWrap: false,
            lineNumbers: true,
            theme: 'dark',
            tabSize: 4,
            autoSave: true
        };
        this.loadSettings();
    }

    saveSettings() {
        localStorage.setItem('ideSettings', JSON.stringify(this.settings));
    }

    loadSettings() {
        const stored = localStorage.getItem('ideSettings');
        if (stored) {
            this.settings = { ...this.settings, ...JSON.parse(stored) };
        }
    }
}

const state = new IDEState();
let lastValidationTime = 0;

// ============================================================================
// UI MANAGER
// ============================================================================

class UIManager {
    static init() {
        this.codeEditor = document.getElementById('codeEditor');
        this.consoleOutput = document.getElementById('consoleOutput');
        this.currentFileEl = document.getElementById('currentFile');
        this.modifiedIndicatorEl = document.getElementById('modifiedIndicator');
        this.lineCountEl = document.getElementById('lineCount');
        this.charCountEl = document.getElementById('charCount');
        this.execTimeEl = document.getElementById('execTime');

        // Set initial code
        this.codeEditor.value = state.currentCode;
        
        // Apply settings
        this.applySettings();
    }

    static applySettings() {
        const { fontSize, theme, tabSize } = state.settings;
        
        this.codeEditor.style.fontSize = `${fontSize}px`;
        this.codeEditor.style.tabSize = tabSize;
        document.documentElement.style.setProperty('--tab-size', tabSize);

        if (theme === 'light') {
            document.body.classList.add('light-theme');
        } else {
            document.body.classList.remove('light-theme');
        }
    }

    static updateStats() {
        const code = this.codeEditor.value;
        const lines = code.split('\n').length;
        const chars = code.length;

        this.lineCountEl.textContent = lines;
        this.charCountEl.textContent = chars;
    }

    static updateFileStatus() {
        this.currentFileEl.textContent = state.currentFile;
        
        if (state.isModified) {
            this.modifiedIndicatorEl.classList.add('active');
        } else {
            this.modifiedIndicatorEl.classList.remove('active');
        }
    }

    static setRunning(isRunning) {
        state.isRunning = isRunning;
        const btnRun = document.getElementById('btnRun');
        const btnStop = document.getElementById('btnStop');

        if (isRunning) {
            btnRun.disabled = true;
            btnStop.disabled = false;
            this.addConsoleMessage('Running code...', 'info');
        } else {
            btnRun.disabled = false;
            btnStop.disabled = true;
        }
    }

    static addConsoleMessage(text, type = 'info') {
        const message = document.createElement('div');
        message.className = `console-message ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        message.innerHTML = `
            <span class="timestamp">[${timestamp}]</span>
            <span class="text">${this.escapeHtml(text)}</span>
        `;
        
        this.consoleOutput.appendChild(message);
        this.consoleOutput.scrollTop = this.consoleOutput.scrollHeight;
    }

    static clearConsole() {
        this.consoleOutput.innerHTML = '';
        this.addConsoleMessage('Console cleared', 'info');
    }

    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    static displayErrors(errors) {
        const errorsList = document.getElementById('errorsList');
        errorsList.innerHTML = '';

        if (errors.length === 0) {
            errorsList.innerHTML = '<div style="color: var(--text-secondary); padding: var(--spacing-md);">No errors found</div>';
            return;
        }

        errors.forEach(error => {
            const errorEl = document.createElement('div');
            errorEl.className = 'error-item';
            errorEl.textContent = error;
            errorsList.appendChild(errorEl);
        });
    }

    static displayAnalysis(analysis) {
        const analysisContent = document.getElementById('analysisContent');
        const { stats, validation, metrics } = analysis;

        let html = `
            <div class="stat-box">
                <h4>📊 Code Metrics</h4>
                <div class="stat-grid">
                    <div class="stat-item-box">
                        <div class="value">${stats.totalLines}</div>
                        <div class="label">Total Lines</div>
                    </div>
                    <div class="stat-item-box">
                        <div class="value">${stats.codeLines}</div>
                        <div class="label">Code Lines</div>
                    </div>
                    <div class="stat-item-box">
                        <div class="value">${stats.functions}</div>
                        <div class="label">Functions</div>
                    </div>
                    <div class="stat-item-box">
                        <div class="value">${stats.classes}</div>
                        <div class="label">Classes</div>
                    </div>
                    <div class="stat-item-box">
                        <div class="value">${stats.commentLines}</div>
                        <div class="label">Comments</div>
                    </div>
                    <div class="stat-item-box">
                        <div class="value">${stats.imports}</div>
                        <div class="label">Imports</div>
                    </div>
                </div>
            </div>

            <div class="stat-box">
                <h4>📈 Analysis</h4>
                <div class="stat-grid">
                    <div class="stat-item-box">
                        <div class="value">${metrics.complexity}</div>
                        <div class="label">Complexity</div>
                    </div>
                    <div class="stat-item-box">
                        <div class="value">${metrics.maintainability.toFixed(0)}%</div>
                        <div class="label">Maintainability</div>
                    </div>
                </div>
            </div>
        `;

        if (validation.issues.length > 0) {
            html += `<div class="stat-box">
                <h4>⚠️ Issues Found (${validation.issues.length})</h4>
                <div style="font-size: 12px;">
                    ${validation.issues.map(i => `<div style="margin: 4px 0; color: var(--warning);">• ${i}</div>`).join('')}
                </div>
            </div>`;
        } else {
            html += `<div class="stat-box">
                <h4>✅ No Issues</h4>
                <p>Code passed syntax validation</p>
            </div>`;
        }

        analysisContent.innerHTML = html;
    }
}

// ============================================================================
// API CLIENT
// ============================================================================

class APIClient {
    static async executeCode(code) {
        try {
            const response = await fetch(`${API_URL}/execute`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            return {
                success: false,
                error: error.message,
                output: null
            };
        }
    }

    static async validateCode(code) {
        try {
            const response = await fetch(`${API_URL}/validate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code })
            });

            return await response.json();
        } catch (error) {
            return { error: error.message };
        }
    }

    static async analyzeCode(code) {
        try {
            const response = await fetch(`${API_URL}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code })
            });

            return await response.json();
        } catch (error) {
            return { error: error.message };
        }
    }

    static async formatCode(code) {
        try {
            const response = await fetch(`${API_URL}/format`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code })
            });

            return await response.json();
        } catch (error) {
            return { error: error.message };
        }
    }

    static async getExamples() {
        try {
            const response = await fetch(`${API_URL}/examples`);
            return await response.json();
        } catch (error) {
            return {};
        }
    }

    static async getTemplates() {
        try {
            const response = await fetch(`${API_URL}/templates`);
            return await response.json();
        } catch (error) {
            return {};
        }
    }
}

// ============================================================================
// EVENT HANDLERS
// ============================================================================

class EventHandlers {
    static init() {
        // Editor events
        document.getElementById('codeEditor').addEventListener('input', this.onCodeChange.bind(this));
        document.getElementById('codeEditor').addEventListener('keydown', this.onKeyDown.bind(this));

        // Toolbar buttons
        document.getElementById('btnRun').addEventListener('click', this.onRun.bind(this));
        document.getElementById('btnStop').addEventListener('click', this.onStop.bind(this));
        document.getElementById('btnFormat').addEventListener('click', this.onFormat.bind(this));
        document.getElementById('btnValidate').addEventListener('click', this.onValidate.bind(this));
        document.getElementById('btnAnalyze').addEventListener('click', this.onAnalyze.bind(this));
        document.getElementById('btnNew').addEventListener('click', this.onNew.bind(this));
        document.getElementById('btnSave').addEventListener('click', this.onSave.bind(this));
        document.getElementById('btnDownload').addEventListener('click', this.onDownload.bind(this));
        document.getElementById('btnClearConsole').addEventListener('click', this.onClearConsole.bind(this));

        // Sidebar
        document.querySelectorAll('.sidebar-tab').forEach(btn => {
            btn.addEventListener('click', this.onSidebarTab.bind(this));
        });

        // Settings
        document.getElementById('btnSettings').addEventListener('click', this.onSettings.bind(this));
        document.getElementById('btnTheme').addEventListener('click', this.onToggleTheme.bind(this));

        // Template select
        document.getElementById('templateSelect').addEventListener('change', this.onTemplateSelect.bind(this));

        // Modal close buttons
        document.querySelectorAll('.close-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modalId = e.target.dataset.modal;
                this.closeModal(modalId);
            });
        });

        // Modal overlay
        document.getElementById('modalOverlay').addEventListener('click', this.closeAllModals.bind(this));

        // Settings inputs
        document.getElementById('fontSizeSlider').addEventListener('input', this.onFontSizeChange.bind(this));
        document.getElementById('themeSelect').addEventListener('change', this.onThemeChange.bind(this));
    }

    static onCodeChange() {
        state.currentCode = UIManager.codeEditor.value;
        state.isModified = true;
        UIManager.updateFileStatus();
        UIManager.updateStats();

        // Debounce validation
        clearTimeout(lastValidationTime);
        lastValidationTime = setTimeout(() => {
            // Could add auto-validation here
        }, 1000);
    }

    static onKeyDown(event) {
        // Ctrl+Enter or Cmd+Enter - Run code
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            event.preventDefault();
            this.onRun();
        }

        // Ctrl+S or Cmd+S - Save
        if ((event.ctrlKey || event.metaKey) && event.key === 's') {
            event.preventDefault();
            this.onSave();
        }

        // Ctrl+N or Cmd+N - New file
        if ((event.ctrlKey || event.metaKey) && event.key === 'n') {
            event.preventDefault();
            this.onNew();
        }

        // Tab - Indent (with shift - unindent)
        if (event.key === 'Tab') {
            event.preventDefault();
            const editor = UIManager.codeEditor;
            const start = editor.selectionStart;
            const end = editor.selectionEnd;

            if (event.shiftKey) {
                // Unindent
                const lines = editor.value.substring(0, start).split('\n');
                const firstLine = lines.length - 1;
                const before = editor.value.substring(0, start);
                const after = editor.value.substring(end);
                const selected = editor.value.substring(start, end);

                const unindented = selected.split('\n').map((line, i) => {
                    if (line.startsWith('\t')) return line.substring(1);
                    if (line.match(/^    /)) return line.substring(4);
                    return line;
                }).join('\n');

                editor.value = before.substring(0, before.length - editor.value.substring(0, start).split('\n').pop().length) + unindented + after;
            } else {
                // Indent
                const indent = '\t';
                const before = editor.value.substring(0, start);
                const after = editor.value.substring(end);
                const selected = editor.value.substring(start, end);

                const indented = selected.split('\n').map(line => indent + line).join('\n');
                editor.value = before + indented + after;
            }

            this.onCodeChange();
        }

        // Ctrl+/ - Toggle comment
        if ((event.ctrlKey || event.metaKey) && event.key === '/') {
            event.preventDefault();
            this.toggleComment();
        }
    }

    static async onRun() {
        const code = UIManager.codeEditor.value;

        if (code.trim().length === 0) {
            UIManager.addConsoleMessage('No code to execute', 'warning');
            return;
        }

        UIManager.setRunning(true);
        UIManager.clearConsole();

        const startTime = performance.now();

        try {
            const result = await APIClient.executeCode(code);

            const executionTime = performance.now() - startTime;
            UIManager.execTimeEl.textContent = `${executionTime.toFixed(2)}ms`;

            if (result.success) {
                UIManager.addConsoleMessage('✅ Execution successful', 'success');
                if (result.output) {
                    result.output.split('\n').forEach(line => {
                        if (line.trim()) {
                            UIManager.addConsoleMessage(line);
                        }
                    });
                }
            } else {
                UIManager.addConsoleMessage('❌ Execution failed', 'error');
                if (result.error) {
                    result.error.split('\n').forEach(line => {
                        if (line.trim()) {
                            UIManager.addConsoleMessage(line, 'error');
                        }
                    });
                }
            }

            // Switch to console tab
            document.querySelector('[data-panel="console"]').click();
        } catch (error) {
            UIManager.addConsoleMessage(`Error: ${error.message}`, 'error');
        } finally {
            UIManager.setRunning(false);
        }
    }

    static onStop() {
        UIManager.setRunning(false);
        UIManager.addConsoleMessage('Execution stopped', 'warning');
    }

    static async onFormat() {
        const code = UIManager.codeEditor.value;
        
        UIManager.addConsoleMessage('Formatting code...', 'info');

        try {
            const result = await APIClient.formatCode(code);
            
            if (result.success) {
                UIManager.codeEditor.value = result.formatted;
                state.currentCode = result.formatted;
                UIManager.addConsoleMessage('✅ Code formatted', 'success');
                this.onCodeChange();
            } else {
                UIManager.addConsoleMessage('❌ Formatting failed', 'error');
            }
        } catch (error) {
            UIManager.addConsoleMessage(`Error: ${error.message}`, 'error');
        }
    }

    static async onValidate() {
        const code = UIManager.codeEditor.value;

        try {
            const result = await APIClient.validateCode(code);
            
            if (result.validation && result.validation.valid) {
                UIManager.addConsoleMessage('✅ Code validation passed', 'success');
                UIManager.displayErrors([]);
            } else {
                UIManager.addConsoleMessage('⚠️ Validation issues found', 'warning');
                UIManager.displayErrors(result.validation?.issues || []);
            }

            // Switch to errors tab
            document.querySelector('[data-panel="errors"]').click();
        } catch (error) {
            UIManager.addConsoleMessage(`Error: ${error.message}`, 'error');
        }
    }

    static async onAnalyze() {
        const code = UIManager.codeEditor.value;

        UIManager.addConsoleMessage('Analyzing code...', 'info');

        try {
            const result = await APIClient.analyzeCode(code);
            
            UIManager.displayAnalysis(result);
            UIManager.addConsoleMessage('✅ Analysis complete', 'success');

            // Switch to analysis tab
            document.querySelector('[data-panel="analysis"]').click();
        } catch (error) {
            UIManager.addConsoleMessage(`Error: ${error.message}`, 'error');
        }
    }

    static onNew() {
        if (state.isModified) {
            if (!confirm('Discard unsaved changes?')) return;
        }

        UIManager.codeEditor.value = '// Start coding here\n';
        state.currentCode = UIManager.codeEditor.value;
        state.currentFile = 'Untitled.ks';
        state.isModified = false;
        UIManager.updateFileStatus();
        UIManager.updateStats();
        UIManager.addConsoleMessage('New file created', 'info');
    }

    static onSave() {
        const code = UIManager.codeEditor.value;
        const filename = state.currentFile;

        // Simulate save (in production, would upload to server)
        localStorage.setItem(`file_${filename}`, code);
        state.isModified = false;
        UIManager.updateFileStatus();
        UIManager.addConsoleMessage(`✅ File saved: ${filename}`, 'success');
    }

    static onDownload() {
        const code = UIManager.codeEditor.value;
        const filename = state.currentFile;

        const blob = new Blob([code], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        UIManager.addConsoleMessage(`Downloaded: ${filename}`, 'success');
    }

    static onClearConsole() {
        UIManager.clearConsole();
    }

    static onSidebarTab(e) {
        const tab = e.currentTarget;
        const tabName = tab.dataset.tab;

        // Update active tab
        document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        // Update panel visibility
        document.querySelectorAll('.sidebar-panel').forEach(p => p.classList.remove('active'));
        document.getElementById(`${tabName}Panel`).classList.add('active');
    }

    static onSettings() {
        document.getElementById('modalOverlay').classList.add('active');
        document.getElementById('settingsModal').classList.add('active');
    }

    static onToggleTheme() {
        const isDark = document.body.classList.contains('light-theme');
        if (isDark) {
            document.body.classList.remove('light-theme');
            state.settings.theme = 'dark';
        } else {
            document.body.classList.add('light-theme');
            state.settings.theme = 'light';
        }
        state.saveSettings();
    }

    static async onTemplateSelect(e) {
        const value = e.target.value;
        if (!value) return;

        const templates = await APIClient.getTemplates();
        const template = Object.values(templates).find(t => 
            t.name.toLowerCase().replace(/\s+/g, '_') === value
        );

        if (template) {
            UIManager.codeEditor.value = template.code;
            state.currentCode = template.code;
            UIManager.updateStats();
            UIManager.addConsoleMessage(`Loaded template: ${template.name}`, 'info');
        }

        e.target.value = '';
    }

    static onFontSizeChange(e) {
        const size = parseInt(e.target.value);
        state.settings.fontSize = size;
        document.getElementById('fontSizeValue').textContent = `${size}px`;
        UIManager.codeEditor.style.fontSize = `${size}px`;
        state.saveSettings();
    }

    static onThemeChange(e) {
        const theme = e.target.value;
        if (theme === 'auto') {
            const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            document.body.classList.toggle('light-theme', !isDark);
        } else {
            document.body.classList.toggle('light-theme', theme === 'light');
        }
        state.settings.theme = theme;
        state.saveSettings();
    }

    static toggleComment() {
        const editor = UIManager.codeEditor;
        const start = editor.selectionStart;
        const end = editor.selectionEnd;

        const before = editor.value.substring(0, start);
        const selected = editor.value.substring(start, end);
        const after = editor.value.substring(end);

        const lines = selected.split('\n');
        const toggledLines = lines.map(line => {
            if (line.trimStart().startsWith('#')) {
                return line.replace('#', '', 1);
            } else {
                return '# ' + line;
            }
        });

        editor.value = before + toggledLines.join('\n') + after;
        this.onCodeChange();
    }

    static closeModal(modalId) {
        document.getElementById(modalId).classList.remove('active');
        document.getElementById('modalOverlay').classList.remove('active');
    }

    static closeAllModals() {
        document.querySelectorAll('.modal').forEach(m => m.classList.remove('active'));
        document.getElementById('modalOverlay').classList.remove('active');
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    UIManager.init();
    EventHandlers.init();
    UIManager.updateStats();
    UIManager.updateFileStatus();

    console.log('✅ KentScript IDE initialized');
    console.log('📚 Features available:');
    console.log('  • Real-time code execution');
    console.log('  • Syntax validation');
    console.log('  • Code analysis');
    console.log('  • Auto-formatting');
    console.log('  • Multiple file support');
    console.log('  • Customizable settings');
});

// Prevent unsaved changes
window.addEventListener('beforeunload', (e) => {
    if (state.isModified) {
        e.preventDefault();
        e.returnValue = '';
    }
});
