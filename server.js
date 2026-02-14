const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(bodyParser.json({ limit: '100mb' }));
app.use(express.static(path.join(__dirname)));

// ============================================================================
// KENTSCRIPT PATH RESOLUTION
// ============================================================================
// Try multiple locations to find kentscript.py
const possiblePaths = [
    path.join(__dirname, 'kentscript.py'),                                    // Same directory
    path.join(__dirname, '..', 'kentscript.py'),                             // Parent directory
    path.join(__dirname, '..', '..', 'kentscript.py'),                       // Two levels up
    '/usr/local/bin/kentscript.py',                                          // System path
    '/usr/bin/kentscript.py',                                                // System path
    path.expandUser('~/kentscript.py'),                                      // Home directory
    process.env.KENTSCRIPT_PATH || '',                                       // Environment variable
];

let KENTSCRIPT_PATH = null;
let pythonInterpreter = 'python3';

// Function to expand user home
function expandUser(filepath) {
    if (filepath[0] === '~') {
        return path.join(os.homedir(), filepath.slice(1));
    }
    return filepath;
}

// Function to check if file exists and is readable
function findKentscript() {
    for (const possiblePath of possiblePaths) {
        if (!possiblePath) continue;
        try {
            const expanded = expandUser(possiblePath);
            if (fs.existsSync(expanded) && fs.accessSync(expanded, fs.constants.R_OK)) {
                console.log(`✅ Found KentScript at: ${expanded}`);
                return expanded;
            }
        } catch (err) {
            // Continue to next path
        }
    }
    return null;
}

// Try to find KentScript
KENTSCRIPT_PATH = findKentscript();

console.log(`
╔════════════════════════════════════════════════════════════╗
║   🚀 KentScript IDE - COMPLETE DOCUMENTATION SITE 🚀      ║
║                                                            ║
║   ✨ Full language documentation                           ║
║   ✨ Interactive code playground                           ║
║   ✨ Real-time code execution                              ║
║   ✨ Dark/Light theme support                              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
`);

if (KENTSCRIPT_PATH) {
    console.log(`✅ KentScript interpreter found at: ${KENTSCRIPT_PATH}`);
} else {
    console.log(`
⚠️  KentScript interpreter NOT found!

Searched locations:
  • ${possiblePaths[0]}
  • ${possiblePaths[1]}
  • ${possiblePaths[2]}
  • ${possiblePaths[3]}
  • ${possiblePaths[4]}
  • ${possiblePaths[5]}

📝 To fix this:

Option 1: Set KENTSCRIPT_PATH environment variable
  export KENTSCRIPT_PATH=/path/to/kentscript.py
  npm start

Option 2: Place kentscript.py in your project directory
  cp kentscript.py ./

Option 3: Create a symbolic link
  ln -s /path/to/kentscript.py ./kentscript.py

Option 4: Use local Python execution (fallback mode)
  Code will run with limited functionality
    `);
}

// ============================================================================
// CODE EXECUTION ENDPOINT
// ============================================================================

app.post('/api/execute', (req, res) => {
    try {
        const { code, timeout = 15000 } = req.body;

        if (!code || code.trim() === '') {
            return res.json({
                success: false,
                output: '',
                error: 'No code provided',
                execTime: 0
            });
        }

        // If KentScript not found, provide helpful error
        if (!KENTSCRIPT_PATH) {
            // Try to use direct Python evaluation as fallback
            return executePythonFallback(code, res);
        }

        const tempFile = path.join(os.tmpdir(), `ks_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.ks`);
        
        try {
            fs.writeFileSync(tempFile, code);
        } catch (writeErr) {
            return res.json({
                success: false,
                output: '',
                error: `Cannot write temp file: ${writeErr.message}`,
                execTime: 0
            });
        }

        const startTime = Date.now();
        let stdout = '';
        let stderr = '';

        const process = spawn(pythonInterpreter, [KENTSCRIPT_PATH, tempFile], {
            timeout: Math.min(timeout, 30000),
            maxBuffer: 50 * 1024 * 1024,
            shell: false
        });

        process.stdout.on('data', (data) => { 
            stdout += data.toString(); 
        });

        process.stderr.on('data', (data) => { 
            stderr += data.toString(); 
        });

        process.on('close', (exitCode) => {
            const execTime = Date.now() - startTime;
            try { fs.unlinkSync(tempFile); } catch (e) {}

            if (exitCode === 0) {
                res.json({
                    success: true,
                    output: stdout || '(no output)',
                    error: null,
                    execTime: execTime
                });
            } else {
                res.json({
                    success: false,
                    output: stdout,
                    error: stderr || `Exit code ${exitCode}`,
                    execTime: execTime
                });
            }
        });

        process.on('error', (err) => {
            try { fs.unlinkSync(tempFile); } catch (e) {}
            
            let errorMsg = err.message;
            
            // Provide helpful error messages
            if (err.code === 'ENOENT') {
                errorMsg = `Python interpreter not found: ${pythonInterpreter}\n\nInstall with:\n  apt-get install python3 (Debian/Ubuntu)\n  brew install python3 (macOS)\n  python.org (Windows)`;
            }
            
            res.json({
                success: false,
                output: '',
                error: errorMsg,
                execTime: Date.now() - startTime
            });
        });

    } catch (error) {
        res.json({
            success: false,
            output: '',
            error: `Server error: ${error.message}`,
            execTime: 0
        });
    }
});

// ============================================================================
// FALLBACK: PYTHON DIRECT EXECUTION
// ============================================================================
// Simple fallback that executes Python code directly (limited functionality)

function executePythonFallback(code, res) {
    const tempFile = path.join(os.tmpdir(), `py_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.py`);
    
    // Convert KentScript to basic Python (very limited)
    const pythonCode = convertKentScriptToPython(code);
    
    try {
        fs.writeFileSync(tempFile, pythonCode);
    } catch (writeErr) {
        return res.json({
            success: false,
            output: '',
            error: `Cannot write temp file: ${writeErr.message}`,
            execTime: 0
        });
    }

    const startTime = Date.now();
    let stdout = '';
    let stderr = '';

    const process = spawn(pythonInterpreter, [tempFile], {
        timeout: 15000,
        maxBuffer: 50 * 1024 * 1024,
        shell: false
    });

    process.stdout.on('data', (data) => { 
        stdout += data.toString(); 
    });

    process.stderr.on('data', (data) => { 
        stderr += data.toString(); 
    });

    process.on('close', (exitCode) => {
        const execTime = Date.now() - startTime;
        try { fs.unlinkSync(tempFile); } catch (e) {}

        if (exitCode === 0) {
            res.json({
                success: true,
                output: stdout || '(no output)',
                error: null,
                execTime: execTime
            });
        } else {
            res.json({
                success: false,
                output: stdout,
                error: stderr || `Exit code ${exitCode}`,
                execTime: execTime
            });
        }
    });

    process.on('error', (err) => {
        try { fs.unlinkSync(tempFile); } catch (e) {}
        res.json({
            success: false,
            output: '',
            error: `⚠️ Fallback mode - Install KentScript for full support.\n\n${err.message}`,
            execTime: Date.now() - startTime
        });
    });
}

// ============================================================================
// KENTSCRIPT TO PYTHON CONVERTER (Basic)
// ============================================================================

function convertKentScriptToPython(code) {
    let python = code;
    
    // Very basic conversions
    python = python.replace(/print\(/g, 'print(');  // Keep print as is
    python = python.replace(/function\s+(\w+)\s*\(/g, 'def $1(');  // Functions
    python = python.replace(/function\s+(\w+)\s*\{/g, 'def $1():');  // Functions without params
    python = python.replace(/\{/g, ':');  // Braces to colons
    python = python.replace(/\};/g, '');  // End of function
    python = python.replace(/var\s+/g, '');  // Remove var
    python = python.replace(/let\s+/g, '');  // Remove let
    python = python.replace(/const\s+/g, '');  // Remove const
    python = python.replace(/->|\s*=>\s*/g, ':');  // Arrow functions
    python = python.replace(/this\./g, 'self.');  // This to self
    
    return python;
}

// ============================================================================
// HEALTH CHECK ENDPOINT
// ============================================================================

app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        kentscript_found: !!KENTSCRIPT_PATH,
        kentscript_path: KENTSCRIPT_PATH || 'not found',
        python_interpreter: pythonInterpreter,
        node_version: process.version,
        timestamp: new Date().toISOString()
    });
});

// ============================================================================
// STATUS ENDPOINT
// ============================================================================

app.get('/api/status', (req, res) => {
    res.json({
        server: 'running',
        features: {
            kentscript: !!KENTSCRIPT_PATH ? 'enabled' : 'disabled (fallback mode)',
            cors: 'enabled',
            documentation: 'complete',
            ide: 'enabled'
        },
        interpreter: pythonInterpreter,
        temp_directory: os.tmpdir(),
        uptime: process.uptime(),
        memory: process.memoryUsage()
    });
});

// ============================================================================
// STARTUP MESSAGE
// ============================================================================

app.listen(PORT, () => {
    const startupMsg = `
✅ Server running on http://localhost:${PORT}
🌐 Open browser: http://localhost:${PORT}

🔥 Features:
   ✓ Complete documentation
   ✓ Interactive IDE playground
   ✓ Real-time code execution
   ${KENTSCRIPT_PATH ? '✓ Full KentScript support' : '⚠️ Fallback Python mode (limited)'}

📊 Server Status:
   • Port: ${PORT}
   • Python: ${pythonInterpreter}
   • KentScript: ${KENTSCRIPT_PATH ? 'Found' : 'Not found'}
   • Temp dir: ${os.tmpdir()}

🛠️ API Endpoints:
   POST /api/execute    - Execute code
   GET  /api/health     - Health check
   GET  /api/status     - Server status

💡 To use full KentScript:
   ${!KENTSCRIPT_PATH ? `
   Option 1: Set environment variable
     export KENTSCRIPT_PATH=/path/to/kentscript.py
     npm start

   Option 2: Place kentscript.py in project directory
     cp kentscript.py ./
     npm start
   ` : '✓ KentScript is properly configured'}

⌨️  Keyboard shortcut in IDE: Ctrl+Enter to run code
    `;
    console.log(startupMsg);
});

// ============================================================================
// ERROR HANDLING
// ============================================================================

process.on('uncaughtException', (err) => {
    console.error('❌ Uncaught Exception:', err.message);
    console.error(err.stack);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
});

module.exports = app;
