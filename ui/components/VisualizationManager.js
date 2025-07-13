/**
 * VisualizationManager - NB Phase 1
 * Coordinates all visualization components and handles API integration
 * 2W12.one aesthetic coordination
 */

class VisualizationManager {
    constructor(options = {}) {
        this.options = {
            apiEndpoint: '/api/audio/analyze-visualization',
            waveformCanvasId: 'waveformCanvas',
            progressCallback: null,
            errorCallback: null,
            completeCallback: null,
            ...options
        };
        
        // Component instances
        this.waveformCanvas = null;
        this.isAnalyzing = false;
        this.currentAnalysis = null;
        
        // Initialize components
        this.initializeComponents();
        
        console.log('🎛️ VisualizationManager initialized');
    }
    
    initializeComponents() {
        // Initialize WaveformCanvas if container exists
        const canvasElement = document.getElementById(this.options.waveformCanvasId);
        if (canvasElement) {
            this.waveformCanvas = new WaveformCanvas(this.options.waveformCanvasId, {
                backgroundColor: '#fafafa',
                waveformColor: '#1a1a1a',
                downbeatColor: '#ff0080',
                rmsColor: '#8e8e8e'
            });
            
            // Setup event listeners
            canvasElement.addEventListener('waveform-seek', (e) => {
                this.onWaveformSeek(e.detail.time);
            });
            
            console.log('✅ WaveformCanvas component ready');
        } else {
            console.warn('⚠️ Waveform canvas element not found');
        }
    }
    
    /**
     * Analyze audio file with visualization
     */
    async analyzeFile(file) {
        if (this.isAnalyzing) {
            console.warn('⚠️ Analysis already in progress');
            return false;
        }
        
        this.isAnalyzing = true;
        
        try {
            console.log(`🎵 Starting NB visualization analysis: ${file.name}`);
            
            // Progress callback
            if (this.options.progressCallback) {
                this.options.progressCallback({
                    status: 'starting',
                    message: 'Preparing AudioFlux analysis...',
                    progress: 0
                });
            }
            
            // Create form data
            const formData = new FormData();
            formData.append('file', file);
            
            // Progress update
            if (this.options.progressCallback) {
                this.options.progressCallback({
                    status: 'uploading',
                    message: 'Uploading and extracting waveform data...',
                    progress: 20
                });
            }
            
            // Make API request
            const response = await fetch(this.options.apiEndpoint, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            // Progress update
            if (this.options.progressCallback) {
                this.options.progressCallback({
                    status: 'processing',
                    message: 'AudioFlux processing complete, building visualization...',
                    progress: 80
                });
            }
            
            // Parse response
            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.detail || 'Analysis failed');
            }
            
            // Store analysis result
            this.currentAnalysis = result;
            
            // Load visualization data
            this.loadVisualization(result);
            
            // Progress complete
            if (this.options.progressCallback) {
                this.options.progressCallback({
                    status: 'complete',
                    message: 'Visualization ready!',
                    progress: 100
                });
            }
            
            // Complete callback
            if (this.options.completeCallback) {
                this.options.completeCallback(result);
            }
            
            console.log('✅ NB visualization analysis complete');
            return result;
            
        } catch (error) {
            console.error('❌ Visualization analysis failed:', error);
            
            if (this.options.errorCallback) {
                this.options.errorCallback(error);
            }
            
            if (this.options.progressCallback) {
                this.options.progressCallback({
                    status: 'error',
                    message: `Analysis failed: ${error.message}`,
                    progress: 0
                });
            }
            
            return false;
            
        } finally {
            this.isAnalyzing = false;
        }
    }
    
    /**
     * Load visualization data into components
     */
    loadVisualization(analysisResult) {
        console.log('🎨 Loading visualization components...');
        
        // Load waveform canvas
        if (this.waveformCanvas && analysisResult.visualization) {
            const success = this.waveformCanvas.loadVisualizationData(analysisResult);
            if (success) {
                console.log('✅ Waveform visualization loaded');
            } else {
                console.error('❌ Failed to load waveform visualization');
            }
        }
        
        // Update metadata displays
        this.updateMetadataDisplays(analysisResult);
        
        // Show visualization container
        this.showVisualization();
    }
    
    /**
     * Update metadata displays
     */
    updateMetadataDisplays(result) {
        const analysis = result.analysis;
        const visualization = result.visualization;
        
        // Update performance badge
        const performanceBadge = document.getElementById('performanceBadge');
        if (performanceBadge && analysis.response_time && visualization.waveform.duration) {
            const realtimeFactor = (visualization.waveform.duration / analysis.response_time).toFixed(1);
            performanceBadge.textContent = `${realtimeFactor}x REALTIME`;
        }
        
        // Update visualization metadata
        const metadataElements = {
            'viz-duration': `${visualization.waveform.duration}s`,
            'viz-downbeats': `${visualization.downbeats.count} downbeats`,
            'viz-waveform-points': `${visualization.waveform.width} points`,
            'viz-sample-rate': `${visualization.waveform.sample_rate}Hz`,
            'viz-compression': `${visualization.metadata?.compression_ratio?.toFixed(1) || 'N/A'}x compression`
        };
        
        Object.entries(metadataElements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }
    
    /**
     * Show visualization interface
     */
    showVisualization() {
        const visualizationContainer = document.getElementById('visualizationContainer');
        if (visualizationContainer) {
            visualizationContainer.style.display = 'block';
            visualizationContainer.scrollIntoView({ 
                behavior: 'smooth',
                block: 'start'
            });
        }
    }
    
    /**
     * Handle waveform seek events
     */
    onWaveformSeek(time) {
        console.log(`🎯 Seeking to ${time.toFixed(2)}s`);
        
        // Update other components with seek time
        // Future: sync with audio player, chord progression display, etc.
        
        // Emit global seek event
        document.dispatchEvent(new CustomEvent('visualization-seek', {
            detail: { time: time }
        }));
    }
    
    /**
     * Get current analysis data
     */
    getCurrentAnalysis() {
        return this.currentAnalysis;
    }
    
    /**
     * Get visualization metadata
     */
    getVisualizationMetadata() {
        if (this.waveformCanvas) {
            return this.waveformCanvas.getMetadata();
        }
        return null;
    }
    
    /**
     * Export visualization as image
     */
    exportVisualization() {
        if (this.waveformCanvas) {
            return this.waveformCanvas.exportImage();
        }
        return null;
    }
    
    /**
     * Reset visualization state
     */
    reset() {
        this.currentAnalysis = null;
        
        if (this.waveformCanvas) {
            this.waveformCanvas.waveformData = null;
            this.waveformCanvas.downbeats = [];
            this.waveformCanvas.render();
        }
        
        // Hide visualization container
        const visualizationContainer = document.getElementById('visualizationContainer');
        if (visualizationContainer) {
            visualizationContainer.style.display = 'none';
        }
        
        console.log('🔄 Visualization state reset');
    }
    
    /**
     * Update component themes/colors
     */
    updateTheme(theme) {
        if (this.waveformCanvas) {
            Object.assign(this.waveformCanvas.options, theme);
            this.waveformCanvas.render();
        }
    }
}

// Export for use in other modules
window.VisualizationManager = VisualizationManager;