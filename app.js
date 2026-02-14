/* ============================================================================
   KENTSCRIPT - COMPLETE APPLICATION
   ============================================================================ */

// Initialize CodeMirror
const editor = CodeMirror.fromTextArea(document.getElementById('codeEditor'), {
    lineNumbers: true,
    lineWrapping: true,
    mode: 'javascript',
    theme: 'dracula',
    indentUnit: 4,
    tabSize: 4,
    indentWithTabs: false,
    autofocus: true,
    styleActiveLine: true,
    matchBrackets: true,
    autoCloseBrackets: true,
    foldGutter: true,
    gutters: ["CodeMirror-linenumbers", "CodeMirror-foldgutter"]
});

const EXAMPLES = {
    hello: 'print("Hello, KentScript!");',
    variables: `var x = 5;
var y = 10;
var name = "Alice";

print("x =", x);
print("y =", y);
print("Sum:", x + y);`,
    functions: `function add(a, b) {
    return a + b;
};

var square = lambda x -> x * x;

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

var colors = ["red", "green", "blue"];
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
    lists: `var nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

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

var data = {"name": "Alice", "age": 25};
var json_str = json.stringify(data);
print("JSON:", json_str);

var text = "secret";
var encrypted = crypto.encrypt(text, "password");
print("Encrypted!");`
};

// ============================================================================
// MENU FUNCTIONALITY
// ============================================================================

const menuToggle = document.getElementById('menuToggle');
const sidebar = document.getElementById('sidebar');
const navMenu = document.getElementById('navMenu');

// Toggle sidebar
menuToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    sidebar.classList.toggle('open');
});

// Close sidebar when clicking outside
document.addEventListener('click', (e) => {
    if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
        sidebar.classList.remove('open');
    }
});

// Close sidebar function
function closeSidebar() {
    sidebar.classList.remove('open');
}

// Close sidebar when nav item clicked
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        sidebar.classList.remove('open');
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
    });
});

// ============================================================================
// THEME TOGGLE
// ============================================================================

const themeToggle = document.getElementById('themeToggle');

themeToggle.addEventListener('click', () => {
    const theme = document.documentElement.getAttribute('data-theme');
    const newTheme = theme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    document.querySelector('.theme-icon').textContent = newTheme === 'light' ? '🌙' : '☀️';
});

// Load saved theme
const savedTheme = localStorage.getItem('theme') || 'light';
document.documentElement.setAttribute('data-theme', savedTheme);
document.querySelector('.theme-icon').textContent = savedTheme === 'light' ? '🌙' : '☀️';

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
}

editor.on('change', updateStats);

async function runCode() {
    const code = editor.getValue();
    if (!code.trim()) {
        setOutput('Please enter code', 'error');
        return;
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
    document.getElementById('output').className = 'output-empty';
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
    navigator.clipboard.writeText(editor.getValue());
    alert('Code copied to clipboard!');
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
