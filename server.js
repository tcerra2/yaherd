const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.static('./'));

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'app.html'));
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok', message: 'YOLO Tracking App is running' });
});

// Start server
app.listen(PORT, () => {
    console.log(`🎯 YOLO Tracking app running on http://localhost:${PORT}`);
    console.log('All processing happens on the client-side!');
});
