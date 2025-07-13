/**
 * WaveformCanvas Component - NB Phase 1
 * AudioFlux-based waveform visualization with Madmom downbeats
 * 2W12.one aesthetic integration
 */

class WaveformCanvas {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        // Configuration
        this.options = {
            backgroundColor: '#fafafa',     // 2w12.one background
            waveformColor: '#1a1a1a',      // 2w12.one text primary
            downbeatColor: '#ff0080',      // 2w12.one accent pink
            rmsColor: '#8e8e8e',           // 2w12.one text secondary
            gridColor: '#f5f5f5',          // 2w12.one bg secondary
            cursorColor: '#777777',        // 2w12.one neutral gray
            ...options
        };
        
        // Data storage
        this.waveformData = null;
        this.downbeats = [];
        this.duration = 0;
        this.currentTime = 0;
        
        // Canvas state
        this.zoom = 1;
        this.offset = 0;
        this.isDragging = false;
        
        // Initialize canvas
        this.setupCanvas();
        this.setupEventListeners();
        
        console.log('🎨 WaveformCanvas initialized with 2W12.one aesthetics');
    }
    
    setupCanvas() {
        // Set canvas size to container
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = 200; // Fixed height for waveform
        
        // High DPI support
        const dpr = window.devicePixelRatio || 1;
        const rect2 = this.canvas.getBoundingClientRect();
        
        this.canvas.width = rect2.width * dpr;
        this.canvas.height = rect2.height * dpr;
        this.ctx.scale(dpr, dpr);
        
        this.canvas.style.width = rect2.width + 'px';
        this.canvas.style.height = rect2.height + 'px';
    }
    
    setupEventListeners() {
        // Mouse events for interaction
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('wheel', (e) => this.onWheel(e));
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', (e) => this.onTouchStart(e));
        this.canvas.addEventListener('touchmove', (e) => this.onTouchMove(e));
        this.canvas.addEventListener('touchend', (e) => this.onTouchEnd(e));
        
        // Resize handling
        window.addEventListener('resize', () => this.onResize());
    }
    
    /**
     * Load visualization data from AudioFlux backend
     */
    loadVisualizationData(data) {
        console.log('📊 Loading AudioFlux visualization data:', data);
        
        if (data.visualization && data.visualization.waveform) {
            this.waveformData = data.visualization.waveform;
            this.downbeats = data.visualization.downbeats?.times || [];
            this.duration = this.waveformData.duration || 0;
            
            console.log(`✅ Loaded: ${this.waveformData.width} waveform points, ${this.downbeats.length} downbeats, ${this.duration}s duration`);
            
            // Render the visualization
            this.render();
            
            return true;
        } else {
            console.error('❌ Invalid visualization data structure');
            return false;
        }
    }
    
    /**
     * Main rendering function
     */
    render() {
        if (!this.waveformData) return;
        
        // Clear canvas
        this.ctx.fillStyle = this.options.backgroundColor;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid
        this.drawGrid();
        
        // Draw waveform
        this.drawWaveform();
        
        // Draw downbeats
        this.drawDownbeats();
        
        // Draw current time cursor
        this.drawCursor();
        
        // Draw time labels
        this.drawTimeLabels();
    }
    
    drawGrid() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.ctx.strokeStyle = this.options.gridColor;
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([2, 4]);
        
        // Horizontal center line
        this.ctx.beginPath();
        this.ctx.moveTo(0, height / 2);
        this.ctx.lineTo(width, height / 2);
        this.ctx.stroke();
        
        // Vertical time grid (every 10 seconds)
        const timeStep = 10; // 10 seconds
        const pixelsPerSecond = width / this.duration;
        
        for (let time = timeStep; time < this.duration; time += timeStep) {
            const x = time * pixelsPerSecond;
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, height);
            this.ctx.stroke();
        }
        
        this.ctx.setLineDash([]);
    }
    
    drawWaveform() {
        if (!this.waveformData.peaks || !this.waveformData.valleys) return;
        
        const width = this.canvas.width;
        const height = this.canvas.height;
        const centerY = height / 2;
        
        const peaks = this.waveformData.peaks;
        const valleys = this.waveformData.valleys;
        const pointsCount = peaks.length;
        
        // Draw waveform path
        this.ctx.strokeStyle = this.options.waveformColor;
        this.ctx.lineWidth = 1;
        this.ctx.fillStyle = this.options.waveformColor + '20'; // 20% opacity fill
        
        this.ctx.beginPath();
        
        // Top waveform (peaks)
        for (let i = 0; i < pointsCount; i++) {
            const x = (i / pointsCount) * width;
            const y = centerY - (peaks[i] * centerY * 0.8); // 80% of center height
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        // Bottom waveform (valleys) - draw in reverse
        for (let i = pointsCount - 1; i >= 0; i--) {
            const x = (i / pointsCount) * width;
            const y = centerY - (valleys[i] * centerY * 0.8);
            this.ctx.lineTo(x, y);
        }
        
        this.ctx.closePath();
        this.ctx.fill();
        this.ctx.stroke();
    }
    
    drawDownbeats() {
        if (!this.downbeats.length) return;
        
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.ctx.strokeStyle = this.options.downbeatColor;
        this.ctx.lineWidth = 2;
        this.ctx.globalAlpha = 0.8;
        
        this.downbeats.forEach((beatTime, index) => {
            const x = (beatTime / this.duration) * width;
            
            // Draw downbeat line
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, height);
            this.ctx.stroke();
            
            // Draw downbeat number (every 4th beat)
            if (index % 4 === 0) {
                this.ctx.fillStyle = this.options.downbeatColor;
                this.ctx.font = '10px JetBrains Mono, monospace';
                this.ctx.fillText(`${index + 1}`, x + 2, 15);
            }
        });
        
        this.ctx.globalAlpha = 1;
    }
    
    drawCursor() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const x = (this.currentTime / this.duration) * width;
        
        this.ctx.strokeStyle = this.options.cursorColor;
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([4, 4]);
        
        this.ctx.beginPath();
        this.ctx.moveTo(x, 0);
        this.ctx.lineTo(x, height);
        this.ctx.stroke();
        
        this.ctx.setLineDash([]);
    }
    
    drawTimeLabels() {
        const width = this.canvas.width;
        
        this.ctx.fillStyle = this.options.rmsColor;
        this.ctx.font = '12px JetBrains Mono, monospace';
        
        // Time labels every 30 seconds
        const timeStep = 30;
        const pixelsPerSecond = width / this.duration;
        
        for (let time = 0; time <= this.duration; time += timeStep) {
            const x = time * pixelsPerSecond;
            const minutes = Math.floor(time / 60);
            const seconds = Math.floor(time % 60);
            const timeLabel = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            this.ctx.fillText(timeLabel, x + 4, this.canvas.height - 8);
        }
    }
    
    // Event handlers
    onMouseDown(e) {
        this.isDragging = true;
        this.seekToPosition(e);
    }
    
    onMouseMove(e) {
        if (this.isDragging) {
            this.seekToPosition(e);
        }
    }
    
    onMouseUp(e) {
        this.isDragging = false;
    }
    
    seekToPosition(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const seekTime = (x / rect.width) * this.duration;
        
        this.setCurrentTime(seekTime);
        
        // Emit seek event
        this.canvas.dispatchEvent(new CustomEvent('waveform-seek', {
            detail: { time: seekTime }
        }));
    }
    
    setCurrentTime(time) {
        this.currentTime = Math.max(0, Math.min(time, this.duration));
        this.render();
    }
    
    onResize() {
        this.setupCanvas();
        if (this.waveformData) {
            this.render();
        }
    }
    
    // Touch events
    onTouchStart(e) {
        e.preventDefault();
        const touch = e.touches[0];
        this.onMouseDown(touch);
    }
    
    onTouchMove(e) {
        e.preventDefault();
        const touch = e.touches[0];
        this.onMouseMove(touch);
    }
    
    onTouchEnd(e) {
        e.preventDefault();
        this.onMouseUp(e);
    }
    
    onWheel(e) {
        e.preventDefault();
        
        // Zoom functionality
        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
        this.zoom = Math.max(1, Math.min(this.zoom * zoomFactor, 10));
        
        this.render();
    }
    
    // Public API
    getMetadata() {
        return {
            duration: this.duration,
            downbeatCount: this.downbeats.length,
            waveformPoints: this.waveformData?.width || 0,
            currentTime: this.currentTime
        };
    }
    
    exportImage() {
        return this.canvas.toDataURL('image/png');
    }
}

// Export for use in other modules
window.WaveformCanvas = WaveformCanvas;