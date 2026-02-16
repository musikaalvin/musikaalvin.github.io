/* ============================================================================
   KENTSCRIPT - COMPLETE APPLICATION - ENHANCED
   ============================================================================ */

// ============================================================================
// CODEMIRROR INITIALIZATION WITH KENTSCRIPT SYNTAX
// ============================================================================

// Define Kentscript language for CodeMirror
CodeMirror.defineMode("kentscript", function(config, parserConfig) {
    const keywords = /^(function|class|var|let|const|if|else|for|while|return|import|export|async|await|function|lambda|match|case|default|new|this|self|static|import|from|as|in|not|and|or|is|None|True|False|extends|implements|try|catch|finally|throw|yield|break|continue|do|switch|interface|enum|type|module)\b/;
    
    const types = /^(String|Int|Float|Bool|List|Dict|Set|Any|Object|Array|Tuple|Optional|Union|Map|Function|Promise|Iterator|Generator|Enum|Type|Interface|Class|Module|Number|Boolean)\b/;
    
    const builtins = /^(print|len|range|str|int|float|bool|list|dict|set|map|filter|reduce|zip|enumerate|sorted|reversed|sum|min|max|abs|round|pow|sqrt|isinstance|type|id|hash|dir|eval|exec|compile|open|input|output|json|crypto|http|math|time|threading|datetime|random|os|sys|file|io|net|gui|db|sql)\b/;

    let tokenBase = function(stream, state) {
        // Whitespace and comments
        if (stream.sol()) state.indent = stream.indentation();
        if (stream.eatSpace()) return null;

        // Comments (:: style)
        if (stream.match('::')) {
            stream.skipToEnd();
            return "comment";
        }

        // Strings
        let ch = stream.next();
        if (ch == '"' || ch == "'") {
            state.tokenize = tokenString(ch);
            return state.tokenize(stream, state);
        }

        // Numbers
        if (/\d/.test(ch)) {
            stream.eatWhile(/[\d_]/);
            if (stream.eat('.')) {
                stream.eatWhile(/[\d_]/);
            }
            return "number";
        }

        // Operators and delimiters
        if (/[\+\-\*\/%&\|^~!<>=]/.test(ch)) {
            stream.eatWhile(/[=<>!+\-*/%&|^~]/);
            return "operator";
        }

        // Identifiers and keywords
        stream.eatWhile(/[\w$_]/);
        let word = stream.current();

        if (keywords.test(word)) {
            return "keyword";
        }
        if (types.test(word)) {
            return "type";
        }
        if (builtins.test(word)) {
            return "builtin";
        }
        
        return "variable";
    };

    function tokenString(quote) {
        return function(stream, state) {
            let escaped = false;
            while (true) {
                let ch = stream.next();
                if (ch == quote && !escaped) break;
                if (ch == null) break;
                escaped = !escaped && ch == '\\';
            }
            if (!escaped) state.tokenize = tokenBase;
            return "string";
        };
    }

    return {
        startState: function() {
            return {tokenize: tokenBase, indent: 0};
        },
        token: function(stream, state) {
            return state.tokenize(stream, state);
        }
    };
});

// Initialize CodeMirror with enhanced settings
const editor = CodeMirror.fromTextArea(document.getElementById('codeEditor'), {
    lineNumbers: true,
    lineWrapping: true,
    mode: 'kentscript',
    theme: 'dracula',
    indentUnit: 4,
    tabSize: 4,
    indentWithTabs: false,
    autofocus: false,
    styleActiveLine: true,
    matchBrackets: true,
    autoCloseBrackets: true,
    foldGutter: true,
    gutters: ["CodeMirror-linenumbers", "CodeMirror-foldgutter"],
    viewportMargin: Infinity
});

// ============================================================================
// ERROR DETECTION
// ============================================================================

function detectErrors(code) {
    // Much more lenient - only catch OBVIOUS errors
    const errors = [];
    const lines = code.split('\n');

    lines.forEach((line, index) => {
        const lineNum = index + 1;
        
        // Skip comments
        const commentIndex = line.indexOf('::');
        const codePart = commentIndex >= 0 ? line.substring(0, commentIndex) : line;
        
        if (codePart.trim().length === 0) return;

        // Only check for OBVIOUS bracket mismatches (excluding comments)
        let bracketBalance = 0;
        for (let char of codePart) {
            if (char === '{' || char === '[' || char === '(') bracketBalance++;
            else if (char === '}' || char === ']' || char === ')') bracketBalance--;
        }
        
        // Only report if severely unbalanced (not just single line)
        if (bracketBalance < -1) {
            errors.push({
                line: lineNum,
                message: 'Unmatched closing bracket',
                type: 'syntax'
            });
        }
    });

    return errors;
}

// ============================================================================
// COPY CODE EXAMPLES TO IDE
// ============================================================================

function copyCodeExample(code) {
    // Unescape the code
    let unescaped = code.replace(/\\n/g, '\n').replace(/\\'/g, "'").replace(/\\\\/g, '\\');
    
    // Set the editor value
    editor.setValue(unescaped);
    
    // Scroll to IDE
    document.getElementById('playground').scrollIntoView({ behavior: 'smooth' });
    
    // Show feedback
    setTimeout(() => {
        alert('✅ Code loaded in IDE! Ready to run.');
    }, 500);
}

// Alternative: Copy to clipboard only
function copyToClipboard(code) {
    let unescaped = code.replace(/\\n/g, '\n').replace(/\\'/g, "'").replace(/\\\\/g, '\\');
    navigator.clipboard.writeText(unescaped).then(() => {
        // Flash feedback on button
        const btn = event.target;
        const original = btn.textContent;
        btn.textContent = '✓ Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = original;
            btn.classList.remove('copied');
        }, 2000);
    });
}

// ============================================================================

const EXAMPLES = {
    hello: 'print("Hello, KentScript!");',
    variables: `let x = 5;
let y = 10;
let name = "Alice";

print("x =", x);
print("y =", y);
print("Sum:", x + y);`,
    functions: `function add(a, b) {
    return a + b;
};

let square = lambda x -> x * x;

print("add(5, 3) =", add(5, 3));
print("square(5) =", square(5));`,
    fibonacci: `function fib(n) {
    if (n <= 1) {
        return n;
    };
    return fib(n - 1) + fib(n - 2);
};

for (i in range(10)) {
    print("fib(" + i + ") =", fib(i));
};`,
    loops: `for (i in range(5)) {
    print("i =", i);
};

let colors = ["red", "green", "blue"];
for (color in colors) {
    print(color);
};`,
    classes: `class Person {
    function __init__(name, age) {
        this.name = name;
        this.age = age;
    };
    
    function greet() {
        print("Hello, I'm " + this.name);
    };
};

let p = new Person("Alice", 25);
p.greet();`,
    lists: `let nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

print("nums[0] =", nums[0]);
print("nums[-1] =", nums[-1]);
print("nums[1:4] =", nums[1:4]);
print("nums[::2] =", nums[::2]);

nums.append(11);
print("After append:", nums);`,
    pattern: `function describe(n) {
    match n {
        case 0: { print("Zero"); }
        case 1: { print("One"); }
        case 2 | 3 | 4: { print("Small"); }
        default: { print("Other"); }
    }
};

describe(0);
describe(1);
describe(5);`,
    modules: `import json;
import crypto;

let data = {"name": "Alice", "age": 25};
let json_str = json.stringify(data);
print("JSON:", json_str);

let text = "secret";
let encrypted = crypto.encrypt(text, "password");
print("Encrypted!");`
};

// ============================================================================
// MENU FUNCTIONALITY - WITH PROPER TOUCH SUPPORT
// ============================================================================

const menuToggle = document.getElementById('menuToggle');
const sidebar = document.getElementById('sidebar');
const navMenu = document.getElementById('navMenu');

// Track if we're in a touch event
let isTouchEvent = false;

// Toggle sidebar on click
menuToggle.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    toggleSidebar();
});

// Toggle sidebar on touch
menuToggle.addEventListener('touchstart', (e) => {
    isTouchEvent = true;
    e.preventDefault();
    e.stopPropagation();
    toggleSidebar();
});

menuToggle.addEventListener('touchend', (e) => {
    e.preventDefault();
    e.stopPropagation();
});

function toggleSidebar() {
    sidebar.classList.toggle('open');
    menuToggle.classList.toggle('active');
}

// Close sidebar when clicking outside
document.addEventListener('click', (e) => {
    if (!isTouchEvent && sidebar.classList.contains('open') && 
        !sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
        sidebar.classList.remove('open');
        menuToggle.classList.remove('active');
    }
    isTouchEvent = false;
});

// Close sidebar when touching outside
document.addEventListener('touchstart', (e) => {
    if (sidebar.classList.contains('open') && 
        !sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
        sidebar.classList.remove('open');
        menuToggle.classList.remove('active');
    }
});

// Close sidebar function
function closeSidebar() {
    sidebar.classList.remove('open');
    menuToggle.classList.remove('active');
}

// Close sidebar when nav item clicked
document.querySelectorAll('.nav-item, .menu-section a').forEach(item => {
    item.addEventListener('click', (e) => {
        closeSidebar();
    });
});

// ============================================================================
// THEME TOGGLE - WITH PROPER TOUCH SUPPORT
// ============================================================================

const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.querySelector('.theme-icon');

// Handle click
themeToggle.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    toggleTheme();
});

// Handle touch
themeToggle.addEventListener('touchstart', (e) => {
    e.preventDefault();
    e.stopPropagation();
});

themeToggle.addEventListener('touchend', (e) => {
    e.preventDefault();
    e.stopPropagation();
    toggleTheme();
});

function toggleTheme() {
    const theme = document.documentElement.getAttribute('data-theme');
    const newTheme = theme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    themeIcon.textContent = theme === 'light' ? '🌙' : '☀️';
}

// Load saved theme
const savedTheme = localStorage.getItem('theme') || 'light';
document.documentElement.setAttribute('data-theme', savedTheme);
updateThemeIcon(savedTheme);

// ============================================================================
// IDE FUNCTIONALITY
// ============================================================================

function loadExample() {
    const select = document.getElementById('examplesSelect');
    if (select.value && EXAMPLES[select.value]) {
        editor.setValue(EXAMPLES[select.value]);
        select.value = '';
        updateStats();
    }
}

function updateStats() {
    const code = editor.getValue();
    const lines = code.split('\n').length;
    const chars = code.length;
    const functions = (code.match(/function\s+\w+/g) || []).length;

    document.getElementById('lineCount').textContent = lines;
    document.getElementById('charCount').textContent = chars;
    document.getElementById('funcCount').textContent = functions;

    // Check for errors
    const errors = detectErrors(code);
    const errorCount = errors.length;
    
    if (errorCount > 0) {
        console.warn('Errors detected:', errors);
    }
}

editor.on('change', updateStats);

async function runCode() {
    const code = editor.getValue();
    if (!code.trim()) {
        setOutput('Please enter code', 'error');
        return;
    }

    // Check for errors but DON'T BLOCK - just log them
    const errors = detectErrors(code);
    if (errors.length > 0) {
        const errorMsg = errors.map(e => `Line ${e.line}: ${e.message} (${e.type})`).join('\n');
        console.warn('⚠️ Potential errors detected:\n' + errorMsg);
    }

    setOutput('⏳ Running code...');
    const startTime = Date.now();

    try {
        const response = await fetch('http://localhost:3000/api/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code, timeout: 15000 })
        });

        const result = await response.json();
        const execTime = Date.now() - startTime;

        if (result.success) {
            setOutput(result.output || '(no output)', 'success');
        } else {
            setOutput(`Error:\n${result.error}`, 'error');
        }

        document.getElementById('execTime').textContent = execTime;
        document.getElementById('status').textContent = result.success ? '✓ Success' : '✗ Error';
    } catch (error) {
        setOutput(`❌ Cannot connect to backend\nStart with: npm start`, 'error');
    }
}

function setOutput(text, type = 'info') {
    const output = document.getElementById('output');
    output.textContent = text;
    output.className = type === 'error' ? 'error' : type === 'success' ? 'success' : '';
}

function clearOutput() {
    document.getElementById('output').textContent = 'Output will appear here...';
    document.getElementById('output').className = '';
}

function clearCode() {
    if (confirm('Clear all code?')) {
        editor.setValue('');
        clearOutput();
        updateStats();
    }
}

function formatCode() {
    const code = editor.getValue();
    editor.setValue(code.split('\n').map(l => l.trim()).join('\n'));
}

function copyCode() {
    navigator.clipboard.writeText(editor.getValue()).then(() => {
        const btn = event.target;
        const originalText = btn.textContent;
        btn.textContent = '✓ Copied!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);
    });
}

// Keyboard shortcut
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        runCode();
    }
});

// Initial setup
editor.setValue(EXAMPLES.hello);
updateStats();

// ============================================================================
// SCROLL & NAV ACTIVE STATE
// ============================================================================

window.addEventListener('scroll', () => {
    let current = '#home';
    
    document.querySelectorAll('section').forEach(section => {
        const sectionTop = section.offsetTop;
        if (pageYOffset >= sectionTop - 200) {
            current = '#' + section.getAttribute('id');
        }
    });
    
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    document.querySelector(`a[href="${current}"].nav-item`)?.classList.add('active');
});

// ============================================================================
// PREVENT DOUBLE TAP ZOOM ON BUTTONS
// ============================================================================

let lastTouchEnd = 0;
document.addEventListener('touchend', function(event) {
    const now = Date.now();
    if (now - lastTouchEnd <= 300) {
        event.preventDefault();
    }
    lastTouchEnd = now;
}, false);

// ============================================================================
// ADD COPY BUTTONS TO ALL CODE EXAMPLES ON PAGE LOAD
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Find all code blocks
    document.querySelectorAll('pre code').forEach((codeBlock, index) => {
        const preElement = codeBlock.parentElement;
        
        // Skip if already has wrapper
        if (preElement.parentElement.classList.contains('code-wrapper')) {
            return;
        }
        
        // Get code content
        const code = codeBlock.textContent;
        
        // Create wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'code-wrapper';
        wrapper.style.position = 'relative';
        
        // Create copy button
        const btn = document.createElement('button');
        btn.className = 'copy-code-btn';
        btn.innerHTML = '📋 Copy to IDE';
        btn.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            loadCodeToIDE(code);
        };
        
        // Insert button before code block
        preElement.parentElement.insertBefore(wrapper, preElement);
        wrapper.appendChild(preElement);
        wrapper.insertBefore(btn, preElement);
    });
});

function loadCodeToIDE(code) {
    // Set editor value
    editor.setValue(code);
    
    // Smooth scroll to IDE
    const playground = document.getElementById('playground');
    if (playground) {
        playground.scrollIntoView({ behavior: 'smooth' });
        
        // Focus on editor
        setTimeout(() => {
            editor.focus();
            alert('✅ Code loaded in IDE! Press Ctrl+Enter to run.');
        }, 500);
    }
}
