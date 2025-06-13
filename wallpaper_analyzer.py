import eventlet
eventlet.monkey_patch()

import os
import logging
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
from flask_cors import CORS
import multiprocessing
from functools import lru_cache
from collections import defaultdict
import time

# Configuration
class Config:
    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else \
             torch.device("cuda") if torch.cuda.is_available() else \
             torch.device("cpu")
    
    BATCH_SIZE = 64 if torch.backends.mps.is_available() else 32 if torch.cuda.is_available() else 8
    MAX_WORKERS = 16 if torch.backends.mps.is_available() else multiprocessing.cpu_count()
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    DEFAULT_SIMILARITY_THRESHOLD = 0.85
    DEFAULT_AESTHETIC_THRESHOLD = 0.8
    
    # Flask settings
    HOST = '0.0.0.0'
    PORT = 8000
    DEBUG = False

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analyzed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('WallpaperAnalyzer')

# Flask app initialization
app = Flask(__name__)
CORS(app)
socketio = SocketIO(
    app,
    async_mode='eventlet',
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8
)

# Global state
analysis_results = {}
active_connections = set()

class ImageProcessor:
    def __init__(self):
        self.device = Config.DEVICE
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize neural network models"""
        logger.info(f"Initializing models on device: {self.device}")
        
        # Feature extraction model
        self.feature_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.feature_model = self.feature_model.to(self.device)
        self.feature_model.eval()
        
        # Aesthetic evaluation model
        self.aesthetic_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.aesthetic_model.fc.in_features
        self.aesthetic_model.fc = torch.nn.Linear(num_features, 1)
        self.aesthetic_model = self.aesthetic_model.to(self.device)
        self.aesthetic_model.eval()

    @lru_cache(maxsize=2000)
    def calculate_hash(self, image_path):
        """Calculate perceptual hash of an image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.feature_model.features(image)
                features = torch.mean(features, dim=[2, 3])
                return features.cpu().numpy().tobytes()
        except Exception as e:
            logger.error(f"Error calculating hash for {image_path}: {e}")
            return None

    def process_batch(self, image_paths, process_type='features'):
        """Process a batch of images for either feature extraction or aesthetic scoring"""
        results = {}
        batch_images = []
        valid_paths = []
        
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                image = self.transform(image)
                batch_images.append(image)
                valid_paths.append(path)
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                continue
        
        if batch_images:
            try:
                batch_tensor = torch.stack(batch_images).to(self.device)
                with torch.no_grad():
                    if process_type == 'features':
                        outputs = self.feature_model.features(batch_tensor)
                        outputs = torch.mean(outputs, dim=[2, 3])
                    else:  # aesthetic scoring
                        outputs = self.aesthetic_model(batch_tensor)
                        outputs = torch.sigmoid(outputs).squeeze()
                    
                    outputs = outputs.cpu().numpy()
                    if outputs.ndim == 0:
                        outputs = np.array([outputs])
                    
                    for path, output in zip(valid_paths, outputs):
                        # Convert NumPy values to Python types
                        if isinstance(output, np.ndarray):
                            results[path] = output.tolist()
                        else:
                            results[path] = float(output)
                        
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
            finally:
                del batch_tensor
                del batch_images
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        
        return results

class DuplicateDetector:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def find_duplicates(self, image_paths, threshold=Config.DEFAULT_SIMILARITY_THRESHOLD, progress_callback=None):
        """Find duplicate images using perceptual hashing and feature comparison"""
        if not image_paths:
            return []

        total_images = len(image_paths)
        logger.info(f"Starting duplicate detection for {total_images} images")
        
        # Phase 1: Hash-based grouping
        image_hashes = defaultdict(list)
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = [executor.submit(self.image_processor.calculate_hash, path) for path in image_paths]
            
            for path, future in zip(image_paths, futures):
                try:
                    hash_value = future.result()
                    if hash_value:
                        image_hashes[hash_value].append(path)
                    if progress_callback:
                        progress_callback(len(image_hashes), total_images, "Calculating image hashes")
                except Exception as e:
                    logger.error(f"Error processing hash: {e}")

        potential_duplicates = [paths for paths in image_hashes.values() if len(paths) > 1]
        
        # Phase 2: Feature-based comparison for remaining images
        if len(image_paths) <= 1000:
            features = {}
            for i in range(0, len(image_paths), Config.BATCH_SIZE):
                batch = image_paths[i:i + Config.BATCH_SIZE]
                batch_features = self.image_processor.process_batch(batch, 'features')
                features.update(batch_features)
                if progress_callback:
                    progress_callback(i + len(batch), total_images, "Extracting image features")
            
            # Calculate similarity matrix
            paths = list(features.keys())
            if paths:
                feature_matrix = np.stack([features[path] for path in paths])
                feature_matrix = feature_matrix / np.linalg.norm(feature_matrix, axis=1, keepdims=True)
                similarity_matrix = np.dot(feature_matrix, feature_matrix.T)
                
                # Find similar images
                processed = set()
                for i in range(len(paths)):
                    if i in processed:
                        continue
                    
                    current_group = {paths[i]}
                    processed.add(i)
                    
                    for j in range(i + 1, len(paths)):
                        if j in processed:
                            continue
                        if similarity_matrix[i, j] > threshold:
                            current_group.add(paths[j])
                            processed.add(j)
                    
                    if len(current_group) > 1:
                        potential_duplicates.append(list(current_group))
                    
                    if progress_callback:
                        progress_callback(i + 1, len(paths), "Finding similar images")

        return potential_duplicates

class WallpaperAnalyzer:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.duplicate_detector = DuplicateDetector(self.image_processor)

    def analyze_directory(self, directory, similarity_threshold=Config.DEFAULT_SIMILARITY_THRESHOLD,
                         aesthetic_threshold=Config.DEFAULT_AESTHETIC_THRESHOLD, recursive=False,
                         skip_duplicates=False, skip_aesthetics=False, limit=0, progress_callback=None):
        """Analyze wallpapers in a directory"""
        try:
            logger.info(f"Starting analysis of directory: {directory}")
            
            # Get image paths
            image_paths = self._get_image_paths(directory, recursive)
            if limit > 0:
                image_paths = image_paths[:limit]
            
            results = {
                'directory': directory,
                'total_images': len(image_paths),
                'processed_images': 0,
                'duplicates': [],
                'aesthetic_scores': {},
                'errors': []
            }

            # Process aesthetics
            if not skip_aesthetics:
                for i in range(0, len(image_paths), Config.BATCH_SIZE):
                    batch = image_paths[i:i + Config.BATCH_SIZE]
                    scores = self.image_processor.process_batch(batch, 'aesthetic')
                    results['aesthetic_scores'].update(scores)
                    if progress_callback:
                        progress_callback(i + len(batch), len(image_paths), "Analyzing image aesthetics")

            # Check for duplicates
            if not skip_duplicates:
                results['duplicates'] = self.duplicate_detector.find_duplicates(
                    image_paths,
                    threshold=similarity_threshold,
                    progress_callback=progress_callback
                )

            results['processed_images'] = len(image_paths)
            return results

        except Exception as e:
            logger.error(f"Error analyzing wallpapers: {e}", exc_info=True)
            return {
                'directory': directory,
                'error': str(e),
                'total_images': 0,
                'processed_images': 0,
                'duplicates': [],
                'aesthetic_scores': {},
                'errors': [str(e)]
            }

    def _get_image_paths(self, directory, recursive=False):
        """Get all image paths in a directory"""
        image_paths = []
        
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    if os.path.splitext(file.lower())[1] in Config.IMAGE_EXTENSIONS:
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if os.path.splitext(file.lower())[1] in Config.IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(directory, file))
        
        return image_paths

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        directory = data.get('directory')
        if not directory or not os.path.exists(directory):
            return jsonify({'error': 'Invalid directory path'}), 400

        def progress_callback(current, total, message=None):
            try:
                progress = (current / total) * 100 if total > 0 else 0
                socketio.emit('progress', {
                    'progress': progress,
                    'message': message,
                    'timestamp': time.time(),
                    'processed_images': current,
                    'total_images': total
                })
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

        analyzer = WallpaperAnalyzer()
        results = analyzer.analyze_directory(
            directory=directory,
            similarity_threshold=float(data.get('similarity_threshold', Config.DEFAULT_SIMILARITY_THRESHOLD)),
            aesthetic_threshold=float(data.get('aesthetic_threshold', Config.DEFAULT_AESTHETIC_THRESHOLD)),
            recursive=data.get('recursive', False),
            skip_duplicates=data.get('skip_duplicates', False),
            skip_aesthetics=data.get('skip_aesthetics', False),
            limit=int(data.get('limit', 0)),
            progress_callback=progress_callback
        )

        if results:
            # Convert NumPy float32 values to regular Python floats
            if 'aesthetic_scores' in results:
                results['aesthetic_scores'] = {k: float(v) for k, v in results['aesthetic_scores'].items()}
            
            analysis_results[directory] = results
            socketio.emit('analysis_complete', results)
            return jsonify(results)
        
        return jsonify({'error': 'No results found'}), 404

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/image')
def serve_image():
    try:
        image_path = request.args.get('path')
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(image_path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<path:directory>')
def get_results(directory):
    try:
        if directory in analysis_results:
            return jsonify(analysis_results[directory])
        return jsonify({'error': 'Results not found'}), 404
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    try:
        active_connections.add(request.sid)
        logger.info(f"Client connected: {request.sid}")
        socketio.emit('connection_status', {'status': 'connected'}, room=request.sid)
    except Exception as e:
        logger.error(f"Error in connect handler: {str(e)}")
        socketio.emit('error', {'message': str(e)}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    try:
        active_connections.discard(request.sid)
        logger.info(f"Client disconnected: {request.sid}")
    except Exception as e:
        logger.error(f"Error in disconnect handler: {str(e)}")

@socketio.on_error()
def error_handler(e):
    logger.error(f"Socket.IO error: {str(e)}")
    socketio.emit('error', {'message': str(e)})

def main():
    """Main entry point for the application"""
    try:
        with app.app_context():
            socketio.run(
                app,
                host=Config.HOST,
                port=Config.PORT,
                debug=Config.DEBUG,
                use_reloader=False,
                allow_unsafe_werkzeug=True
            )
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 