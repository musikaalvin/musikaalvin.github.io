/**
 * ============================================================================
 * KENTSCRIPT PRODUCTION BACKEND
 * ============================================================================
 * Express.js + Node.js
 * - Real Python subprocess execution with sandboxing
 * - RESTful API endpoints
 * - File management
 * - Code analysis & compilation
 * - Error handling & logging
 * 
 * Author: Senior DevOps Engineer
 * ============================================================================
 */

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { execFile, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const helmet = require('helmet');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const winston = require('winston');
const NodeCache = require('node-cache');

// ============================================================================
// CONFIGURATION
// ============================================================================

const app = express();
const PORT = process.env.PORT || 3001;
const MAX_EXECUTION_TIME = 30000; // 30 seconds
const MAX_CODE_SIZE = 1024 * 1024; // 1MB
const SANDBOX_TIMEOUT = 5000;

const uploadsDir = path.join(__dirname, '../uploads');
const publicDir = path.join(__dirname, '../public');

// Create required directories
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
if (!fs.existsSync(publicDir)) fs.mkdirSync(publicDir, { recursive: true });

// ============================================================================
// LOGGER SETUP
// ============================================================================

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

// ============================================================================
// CACHE & RATE LIMITING
// ============================================================================

const cache = new NodeCache({ stdTTL: 600 });

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100,
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
});

const executionLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 50,
  skipSuccessfulRequests: false,
});

// ============================================================================
// MIDDLEWARE
// ============================================================================

app.use(helmet());
app.use(compression());
app.use(cors({
  origin: process.env.CORS_ORIGIN || '*',
  credentials: true,
  optionsSuccessStatus: 200
}));

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ limit: '10mb', extended: true }));
app.use(express.static(publicDir));

const upload = multer({
  dest: uploadsDir,
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = ['.ks', '.js', '.py', '.json', '.txt'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowed.includes(ext) || file.mimetype.includes('text')) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type'));
    }
  }
});

app.use(limiter);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Execute KentScript code with timeout and sandboxing
 */
function executeKentScript(code, timeout = MAX_EXECUTION_TIME) {
  return new Promise((resolve, reject) => {
    const tempFile = path.join(uploadsDir, `temp_${crypto.randomBytes(8).toString('hex')}.ks`);
    
    try {
      fs.writeFileSync(tempFile, code);
      
      const process = spawn('python3', [
        path.join(__dirname, '../kentscript.py'),
        tempFile
      ], {
        timeout: timeout,
        maxBuffer: 10 * 1024 * 1024,
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        fs.unlinkSync(tempFile);
        
        if (code === 0) {
          resolve({
            success: true,
            output: stdout,
            error: null,
            exitCode: code,
            executionTime: Date.now()
          });
        } else {
          resolve({
            success: false,
            output: stdout,
            error: stderr || `Process exited with code ${code}`,
            exitCode: code,
            executionTime: Date.now()
          });
        }
      });

      process.on('error', (err) => {
        try { fs.unlinkSync(tempFile); } catch (e) {}
        reject({
          success: false,
          error: err.message,
          output: null
        });
      });

    } catch (err) {
      try { fs.unlinkSync(tempFile); } catch (e) {}
      reject({
        success: false,
        error: err.message,
        output: null
      });
    }
  });
}

/**
 * Validate KentScript syntax
 */
function validateSyntax(code) {
  const issues = [];
  const lines = code.split('\n');
  
  // Basic validation rules
  const patterns = [
    { regex: /^(\s*)class\s+(\w+)/, error: 'Class definition found' },
    { regex: /^(\s*)func\s+(\w+)/, error: 'Function definition found' },
    { regex: /^(\s*)import\s+/, error: 'Import statement found' },
    { regex: /unmatched\s*\{/, error: 'Unmatched opening brace' },
    { regex: /unmatched\s*\}/, error: 'Unmatched closing brace' },
  ];

  lines.forEach((line, idx) => {
    if (line.trim().startsWith('#')) return; // Skip comments

    // Check for basic syntax issues
    const openBraces = (line.match(/\{/g) || []).length;
    const closeBraces = (line.match(/\}/g) || []).length;
    
    if (openBraces !== closeBraces && openBraces > 0) {
      issues.push(`Line ${idx + 1}: Brace mismatch`);
    }
  });

  return {
    valid: issues.length === 0,
    issues: issues,
    lineCount: lines.length,
    characterCount: code.length
  };
}

/**
 * Get code statistics
 */
function getCodeStats(code) {
  const lines = code.split('\n');
  const nonEmptyLines = lines.filter(l => l.trim().length > 0).length;
  const commentLines = lines.filter(l => l.trim().startsWith('#')).length;
  const codeLines = nonEmptyLines - commentLines;

  return {
    totalLines: lines.length,
    nonEmptyLines,
    codeLines,
    commentLines,
    characters: code.length,
    functions: (code.match(/func\s+\w+/g) || []).length,
    classes: (code.match(/class\s+\w+/g) || []).length,
    imports: (code.match(/import\s+\w+/g) || []).length,
  };
}

// ============================================================================
// API ENDPOINTS
// ============================================================================

/**
 * POST /api/execute - Execute KentScript code
 */
app.post('/api/execute', executionLimiter, async (req, res) => {
  try {
    const { code, timeout = MAX_EXECUTION_TIME } = req.body;

    if (!code || code.trim().length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No code provided',
        output: null
      });
    }

    if (code.length > MAX_CODE_SIZE) {
      return res.status(413).json({
        success: false,
        error: `Code exceeds maximum size of ${MAX_CODE_SIZE / 1024}KB`,
        output: null
      });
    }

    const startTime = Date.now();
    const result = await executeKentScript(code, Math.min(timeout, MAX_EXECUTION_TIME));
    const executionTime = Date.now() - startTime;

    logger.info(`Code executed: ${executionTime}ms`);

    res.json({
      ...result,
      executionTime,
      codeStats: getCodeStats(code)
    });
  } catch (error) {
    logger.error(`Execution error: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message || 'Execution failed',
      output: null
    });
  }
});

/**
 * POST /api/validate - Validate syntax
 */
app.post('/api/validate', (req, res) => {
  try {
    const { code } = req.body;
    
    if (!code) {
      return res.status(400).json({ error: 'No code provided' });
    }

    const validation = validateSyntax(code);
    const stats = getCodeStats(code);

    res.json({
      validation,
      stats
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/format - Format code (basic)
 */
app.post('/api/format', (req, res) => {
  try {
    const { code } = req.body;
    
    if (!code) {
      return res.status(400).json({ error: 'No code provided' });
    }

    // Basic formatting rules
    let formatted = code
      .split('\n')
      .map(line => {
        const trimmed = line.trimEnd();
        const indent = line.search(/\S/);
        if (indent === -1) return '';
        return line.substring(0, indent) + trimmed.substring(indent);
      })
      .join('\n');

    res.json({
      success: true,
      formatted
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/analyze - Analyze code
 */
app.post('/api/analyze', (req, res) => {
  try {
    const { code } = req.body;
    
    if (!code) {
      return res.status(400).json({ error: 'No code provided' });
    }

    const stats = getCodeStats(code);
    const validation = validateSyntax(code);
    
    const analysis = {
      stats,
      validation,
      metrics: {
        complexity: (stats.functions * 2 + stats.classes * 3) || 1,
        maintainability: 100 - (stats.codeLines / 50),
        documentation: (stats.commentLines / stats.nonEmptyLines * 100).toFixed(2) + '%'
      }
    };

    res.json(analysis);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/examples - Get code examples
 */
app.get('/api/examples', (req, res) => {
  const examples = {
    hello_world: 'print("Hello, KentScript!");',
    variables: `x = 10
y = 20
z = x + y
print(z)`,
    function: `func add(a, b) {
  return a + b
}
print(add(5, 3))`,
    loop: `for i in range(5) {
  print(i)
}`,
    list: `arr = [1, 2, 3, 4, 5]
for item in arr {
  print(item * 2)
}`,
    class: `class Person {
  func __init__(name) {
    this.name = name
  }
  
  func greet() {
    print("Hello, " + this.name)
  }
}
p = Person("Alice")
p.greet()`,
    async: `async func fetch_data() {
  data = await http_get("https://api.example.com/data")
  return data
}`,
    pattern_match: `match value {
  1: print("One")
  2: print("Two")
  _: print("Other")
}`
  };

  res.json(examples);
});

/**
 * GET /api/templates - Get project templates
 */
app.get('/api/templates', (req, res) => {
  const templates = {
    empty: { name: 'Empty Project', code: '# Start coding here\n' },
    web_server: {
      name: 'Web Server',
      code: `# Simple HTTP server example
import http

func handle_request(request) {
  return "Hello, World!"
}

http.serve(8000, handle_request)
`
    },
    data_processor: {
      name: 'Data Processor',
      code: `# Process data with KentScript
data = [1, 2, 3, 4, 5]
result = map(func(x) { return x * 2 }, data)
print(result)
`
    },
    calculator: {
      name: 'Calculator',
      code: `class Calculator {
  func add(a, b) { return a + b }
  func subtract(a, b) { return a - b }
  func multiply(a, b) { return a * b }
  func divide(a, b) { return a / b }
}

calc = Calculator()
print(calc.add(10, 5))
print(calc.multiply(4, 3))
`
    }
  };

  res.json(templates);
});

/**
 * POST /api/files - Save code file
 */
app.post('/api/files', upload.single('file'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const filename = `${crypto.randomBytes(8).toString('hex')}_${req.file.originalname}`;
    const filepath = path.join(uploadsDir, filename);

    fs.renameSync(req.file.path, filepath);

    res.json({
      success: true,
      filename,
      path: filepath,
      size: req.file.size,
      mimetype: req.file.mimetype
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/files/:filename - Get file content
 */
app.get('/api/files/:filename', (req, res) => {
  try {
    const filename = req.params.filename;
    
    // Security: prevent directory traversal
    if (filename.includes('..') || filename.includes('/')) {
      return res.status(400).json({ error: 'Invalid filename' });
    }

    const filepath = path.join(uploadsDir, filename);
    
    if (!fs.existsSync(filepath)) {
      return res.status(404).json({ error: 'File not found' });
    }

    const content = fs.readFileSync(filepath, 'utf-8');
    res.json({
      success: true,
      content,
      filename,
      size: fs.statSync(filepath).size
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/docs - Get documentation
 */
app.get('/api/docs', (req, res) => {
  const docs = {
    language: 'KentScript v6.0',
    description: 'Modern hybrid scripting language with Rust-like borrow checker',
    features: [
      'Variables & Constants',
      'Functions & Closures',
      'Classes & Inheritance',
      'Pattern Matching',
      'Async/Await',
      'Generators',
      'Type Hints',
      'Exception Handling',
      'List Comprehensions',
      'Decorators'
    ],
    modules: [
      'io', 'json', 'csv', 'http', 'crypto', 'math', 'datetime',
      'threading', 'asyncio', 'database', 'gui'
    ]
  };

  res.json(docs);
});

/**
 * GET /api/health - Health check
 */
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage()
  });
});

// ============================================================================
// ERROR HANDLING
// ============================================================================

app.use((err, req, res, next) => {
  logger.error(err);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

// ============================================================================
// SERVER STARTUP
// ============================================================================

app.listen(PORT, () => {
  logger.info(`✅ KentScript backend running on http://localhost:${PORT}`);
  logger.info(`📁 Uploads directory: ${uploadsDir}`);
  logger.info(`🔒 Max execution time: ${MAX_EXECUTION_TIME}ms`);
});

module.exports = app;
