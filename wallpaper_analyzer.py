# Wallpaper Analyzer V2
# Concurrency: Uses eventlet (green threads) for async I/O and Flask-SocketIO, and ThreadPoolExecutor for CPU-bound image processing. This hybrid model is robust for I/O and parallel CPU workloads.
# Configuration: All magic numbers and tunables are in the Config class below.

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
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO
from flask_cors import CORS
import multiprocessing
from functools import lru_cache
from collections import defaultdict
import sqlite3
import json
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import flask

# Configuration
class Config:
    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else \
             torch.device("cuda") if torch.cuda.is_available() else \
             torch.device("cpu")
    
    BATCH_SIZE = 128 if torch.backends.mps.is_available() else 64 if torch.cuda.is_available() else 16
    MAX_WORKERS = 64 if torch.backends.mps.is_available() else multiprocessing.cpu_count()
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    DEFAULT_SIMILARITY_THRESHOLD = 0.85
    DEFAULT_AESTHETIC_THRESHOLD = 0.8
    
    # Clustering settings
    MIN_CLUSTERS = 5
    MAX_CLUSTERS = 10
    CLUSTER_FEATURE_DIM = 2048  # ResNet50 feature dimension
    
    # Flask settings
    HOST = '0.0.0.0'
    PORT = 8000
    DEBUG = False
    
    # Logging settings
    ENABLE_APP_LOGGING = True  # Application logs
    ENABLE_SOCKET_LOGGING = False  # Socket.IO/Engine.IO logs
    
    # Database settings
    DB_PATH = 'analysis_cache.db'
    # Magic numbers moved here
    MAX_HASH_CACHE = 2000
    DUPLICATE_FEATURE_LIMIT = 1000

# Database initialization
def init_db() -> None:
    conn = sqlite3.connect(Config.DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            directory TEXT PRIMARY KEY,
            similarity_threshold REAL,
            aesthetic_threshold REAL,
            recursive BOOLEAN,
            skip_duplicates BOOLEAN,
            skip_aesthetics BOOLEAN,
            limit_count INTEGER,
            total_images INTEGER,
            processed_images INTEGER,
            duplicates TEXT,
            aesthetic_scores TEXT,
            clusters TEXT,
            cluster_features TEXT,
            last_modified TIMESTAMP,
            last_analyzed TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

def get_cached_results(directory: str, params: dict) -> dict | None:
    """Get cached results if they exist and match the parameters"""
    conn = sqlite3.connect(Config.DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM analysis_results 
        WHERE directory = ? 
        AND similarity_threshold = ?
        AND aesthetic_threshold = ?
        AND recursive = ?
        AND skip_duplicates = ?
        AND skip_aesthetics = ?
        AND limit_count = ?
    ''', (
        directory,
        params['similarity_threshold'],
        params['aesthetic_threshold'],
        params['recursive'],
        params['skip_duplicates'],
        params['skip_aesthetics'],
        params['limit']
    ))
    
    result = c.fetchone()
    conn.close()
    
    if result:
        # Check if directory contents have changed
        last_modified = datetime.fromisoformat(result[13])
        current_modified = datetime.fromtimestamp(os.path.getmtime(directory))
        
        if current_modified <= last_modified:
            # Convert stored JSON strings back to Python objects
            return {
                'directory': result[0],
                'total_images': result[7],
                'processed_images': result[8],
                'duplicates': json.loads(result[9]),
                'aesthetic_scores': json.loads(result[10]),
                'clusters': json.loads(result[11]),
                'cluster_features': json.loads(result[12]),
                'cached': True
            }
    
    return None

def cache_results(directory: str, params: dict, results: dict) -> None:
    """Cache analysis results in the database"""
    conn = sqlite3.connect(Config.DB_PATH)
    c = conn.cursor()
    
    # Get directory's last modified time
    last_modified = datetime.fromtimestamp(os.path.getmtime(directory))
    
    c.execute('''
        INSERT OR REPLACE INTO analysis_results 
        (directory, similarity_threshold, aesthetic_threshold, recursive, 
         skip_duplicates, skip_aesthetics, limit_count, total_images, 
         processed_images, duplicates, aesthetic_scores, clusters, cluster_features,
         last_modified, last_analyzed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        directory,
        params['similarity_threshold'],
        params['aesthetic_threshold'],
        params['recursive'],
        params['skip_duplicates'],
        params['skip_aesthetics'],
        params['limit'],
        results['total_images'],
        results['processed_images'],
        json.dumps(results['duplicates']),
        json.dumps(results['aesthetic_scores']),
        json.dumps(results.get('clusters', {})),
        json.dumps(results.get('cluster_features', {})),
        last_modified.isoformat(),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()

# Logging setup
handlers = [logging.FileHandler('analyzed.log')]
if Config.ENABLE_APP_LOGGING:
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger('WallpaperAnalyzer')

# Flask app initialization
app = Flask(__name__)
CORS(app)
socketio = SocketIO(
    app,
    async_mode='eventlet',
    cors_allowed_origins="*",
    logger=Config.ENABLE_SOCKET_LOGGING,
    engineio_logger=Config.ENABLE_SOCKET_LOGGING,
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8
)

# Global state
analysis_results = {}
active_connections = set()

class ImageProcessor:
    def __init__(self) -> None:
        self.device = Config.DEVICE
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._initialize_models()
        
    def _initialize_models(self) -> None:
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

    @lru_cache(maxsize=Config.MAX_HASH_CACHE)
    def calculate_hash(self, image_path: str) -> bytes | None:
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

    def process_batch(self, image_paths: list[str], process_type: str = 'features', progress_callback: callable = None) -> dict:
        results = {}
        batch_size = Config.BATCH_SIZE
        total = len(image_paths)
        for i in range(0, total, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size+1} ({len(batch_paths)} images) [{i+1}-{min(i+batch_size, total)} / {total}] for {process_type}")
            batch_images = []
            valid_paths = []
            
            for path in batch_paths:
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
            
            if progress_callback:
                progress_callback(min(i+batch_size, total), total)
        
        logger.info(f"Completed {process_type} for {total} images.")
        return results

class DuplicateDetector:
    def __init__(self, image_processor: ImageProcessor) -> None:
        self.image_processor = image_processor

    def find_duplicates(self, image_paths: list[str], threshold: float = Config.DEFAULT_SIMILARITY_THRESHOLD) -> list[list[str]]:
        """Find duplicate images using perceptual hashing and feature comparison"""
        if not image_paths:
            return []

        # Phase 1: Hash-based grouping
        image_hashes = defaultdict(list)
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = [executor.submit(self.image_processor.calculate_hash, path) for path in image_paths]
            
            for path, future in zip(image_paths, futures):
                try:
                    hash_value = future.result()
                    if hash_value:
                        image_hashes[hash_value].append(path)
                except Exception as e:
                    logger.error(f"Error processing hash: {e}")

        potential_duplicates = [paths for paths in image_hashes.values() if len(paths) > 1]
        
        # Phase 2: Feature-based comparison for remaining images
        if len(image_paths) <= Config.DUPLICATE_FEATURE_LIMIT:
            features = {}
            for i in range(0, len(image_paths), Config.BATCH_SIZE):
                batch = image_paths[i:i + Config.BATCH_SIZE]
                batch_features = self.image_processor.process_batch(batch, 'features')
                features.update(batch_features)
            
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

        return potential_duplicates

class ImageClusterer:
    def __init__(self, image_processor: ImageProcessor) -> None:
        self.image_processor = image_processor
        self.device = Config.DEVICE
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize ResNet50 model for feature extraction"""
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final classification layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, image_paths: list[str]) -> tuple[np.ndarray, list[str]]:
        """Extract features from a batch of images"""
        features = []
        valid_paths = []
        
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                image = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feature = self.model(image)
                    feature = feature.squeeze().cpu().numpy()
                    features.append(feature)
                    valid_paths.append(path)
            except Exception as e:
                logger.error(f"Error extracting features for {path}: {e}")
                continue
        
        return np.array(features), valid_paths
    
    def find_optimal_clusters(self, features: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette analysis"""
        best_score = -1
        best_n_clusters = Config.MIN_CLUSTERS
        
        for n_clusters in range(Config.MIN_CLUSTERS, min(Config.MAX_CLUSTERS + 1, len(features))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(features, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
        
        return best_n_clusters
    
    def cluster_images(self, image_paths: list[str]) -> tuple[dict, dict]:
        """Cluster images based on their features"""
        features, valid_paths = self.extract_features(image_paths)
        
        if len(features) < Config.MIN_CLUSTERS:
            return {0: {"paths": valid_paths, "size": len(valid_paths)}}, {}
        
        n_clusters = self.find_optimal_clusters(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Create cluster information
        clusters = {}
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_paths = [valid_paths[idx] for idx in cluster_indices]
            cluster_features = features[cluster_indices]
            
            # Find representative image (closest to cluster center)
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_features - center, axis=1)
            representative_idx = np.argmin(distances)
            
            clusters[i] = {
                "size": len(cluster_paths),
                "representative": cluster_paths[representative_idx],
                "paths": cluster_paths
            }
        
        return clusters, {}

class WallpaperAnalyzer:
    def __init__(self) -> None:
        self.image_processor = ImageProcessor()
        self.duplicate_detector = DuplicateDetector(self.image_processor)
        self.image_clusterer = ImageClusterer(self.image_processor)

    def analyze_directory(self, directory: str, similarity_threshold: float = Config.DEFAULT_SIMILARITY_THRESHOLD,
                         aesthetic_threshold: float = Config.DEFAULT_AESTHETIC_THRESHOLD, recursive: bool = False,
                         skip_duplicates: bool = False, skip_aesthetics: bool = False, limit: int = 0) -> dict:
        import time
        t0 = time.time()
        total_steps = 5
        current_step = 0
        logger.info(f"Starting analysis for {directory}")
        # 1. Load image paths
        logger.info("Loading image paths...")
        image_paths = self._get_image_paths(directory, recursive)
        if limit and limit > 0:
            image_paths = image_paths[:limit]
        logger.info(f"Loaded {len(image_paths)} image paths.")
        current_step += 1
        socketio.emit("analysis_progress", {"progress": int(current_step/total_steps*100), "stage": "Loaded images"})
        t1 = time.time()
        # 2. Feature extraction
        logger.info("Starting feature extraction...")
        features = None
        if not skip_aesthetics or not skip_duplicates:
            features = self.image_processor.process_batch(
                image_paths,
                process_type='features',
                progress_callback=lambda done, total: socketio.emit(
                    "analysis_progress",
                    {"progress": int((current_step + done/total)/total_steps*100), "stage": "Feature extraction"}
                )
            )
        logger.info("Feature extraction complete.")
        current_step += 1
        socketio.emit("analysis_progress", {"progress": int(current_step/total_steps*100), "stage": "Feature extraction complete"})
        t2 = time.time()
        # 3. Duplicate detection
        logger.info("Starting duplicate detection...")
        duplicates = []
        if not skip_duplicates:
            duplicates = self.duplicate_detector.find_duplicates(image_paths, threshold=similarity_threshold)
        logger.info("Duplicate detection complete.")
        current_step += 1
        socketio.emit("analysis_progress", {"progress": int(current_step/total_steps*100), "stage": "Duplicate detection complete"})
        t3 = time.time()
        # 4. Clustering
        logger.info("Starting clustering...")
        clusters = {}
        cluster_features = {}
        if not skip_aesthetics:
            cluster_result = self.image_clusterer.cluster_images(image_paths)
            if isinstance(cluster_result, tuple):
                clusters, cluster_features = cluster_result
            else:
                clusters = cluster_result
        logger.info("Clustering complete.")
        current_step += 1
        socketio.emit("analysis_progress", {"progress": int(current_step/total_steps*100), "stage": "Clustering complete"})
        t4 = time.time()
        # 5. Aesthetic scoring
        logger.info("Starting aesthetic scoring...")
        aesthetic_scores = {}
        if not skip_aesthetics:
            # Compute aesthetic scores for all images
            try:
                scores = self.image_processor.process_batch(
                    image_paths,
                    process_type='aesthetic',
                    progress_callback=lambda done, total: socketio.emit(
                        "analysis_progress",
                        {"progress": int((current_step + done/total)/total_steps*100), "stage": "Aesthetic scoring"}
                    )
                )
                # scores is a dict: {path: score}
                for path, score in scores.items():
                    # Clamp score to [0, 1] and convert to float
                    try:
                        s = float(score)
                        s = max(0.0, min(1.0, s))
                        aesthetic_scores[path] = s
                    except Exception:
                        continue
            except Exception as e:
                logger.error(f"Error during aesthetic scoring: {e}")
        logger.info("Aesthetic scoring complete.")
        current_step += 1
        socketio.emit("analysis_progress", {"progress": int(current_step/total_steps*100), "stage": "Aesthetic scoring complete"})
        t5 = time.time()
        logger.info(f"Timing: image loading: {t1-t0:.2f}s, feature extraction: {t2-t1:.2f}s, duplicate detection: {t3-t2:.2f}s, clustering: {t4-t3:.2f}s, total: {t5-t0:.2f}s")
        # Always return a results dictionary
        return {
            'directory': directory,
            'total_images': len(image_paths),
            'processed_images': len(image_paths),
            'duplicates': duplicates,
            'aesthetic_scores': aesthetic_scores,
            'clusters': clusters,
            'cluster_features': cluster_features,
            'low_aesthetic': [],
        }

    def _get_image_paths(self, directory: str, recursive: bool = False) -> list[str]:
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

    def format_results_for_frontend(self, results: dict) -> dict:
        """Format analysis results for frontend display"""
        try:
            # Initialize sets for tracking
            all_paths = set()
            duplicate_paths = set()
            low_aesthetic_paths = set()
            
            # Process clusters
            clusters = []
            if results.get('clusters'):
                for cluster_id, cluster_data in results['clusters'].items():
                    if isinstance(cluster_data, dict) and 'paths' in cluster_data:
                        cluster_paths = set(cluster_data['paths'])
                        all_paths.update(cluster_paths)
                        clusters.append({
                            'id': cluster_id,
                            'paths': list(cluster_paths),
                            'size': len(cluster_paths)
                        })
            
            # Process duplicates
            if results.get('duplicates'):
                for group in results['duplicates']:
                    if isinstance(group, list):
                        duplicate_paths.update(group)
                        all_paths.update(group)  # Add duplicate paths to all_paths
            
            # Process low aesthetic images
            if results.get('low_aesthetic'):
                if isinstance(results['low_aesthetic'], list):
                    low_aesthetic_paths.update(results['low_aesthetic'])
                    all_paths.update(results['low_aesthetic'])  # Add low aesthetic paths to all_paths
            
            # If no paths were found in clusters/duplicates/low_aesthetic, use all processed images
            if not all_paths and results.get('processed_images', 0) > 0:
                # Get all image paths from the directory
                image_paths = self._get_image_paths(results.get('directory', ''), results.get('recursive', False))
                all_paths.update(image_paths)
            
            # Create final results
            formatted_results = {
                'images': [],
                'clusters': clusters,
                'duplicates': list(duplicate_paths),
                'low_aesthetic': list(low_aesthetic_paths)
            }
            
            # Add all images with their metadata
            for path in all_paths:
                image_data = {
                    'path': path,
                    'cluster': next((c['id'] for c in clusters if path in c['paths']), -1),
                    'cluster_size': next((c['size'] for c in clusters if path in c['paths']), 0),
                    'is_duplicate': path in duplicate_paths,
                    'is_low_aesthetic': path in low_aesthetic_paths,
                    'aesthetic_score': results.get('aesthetic_scores', {}).get(path)
                }
                formatted_results['images'].append(image_data)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error formatting results: {str(e)}")
            return {
                'images': [],
                'clusters': [],
                'duplicates': [],
                'low_aesthetic': []
            }

# Flask routes
@app.route('/')
def index() -> 'flask.Response':
    return send_file('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze() -> 'flask.Response':
    try:
        data = request.get_json()
        directory = data.get('directory')
        logger.info(f"Starting analysis for directory: {directory}")
        
        params = {
            'similarity_threshold': float(data.get('similarity_threshold', Config.DEFAULT_SIMILARITY_THRESHOLD)),
            'aesthetic_threshold': float(data.get('aesthetic_threshold', Config.DEFAULT_AESTHETIC_THRESHOLD)),
            'recursive': bool(data.get('recursive', True)),
            'skip_duplicates': bool(data.get('skip_duplicates', True)),
            'skip_aesthetics': bool(data.get('skip_aesthetics', True)),
            'limit': int(data.get('limit', 0))
        }
        logger.info(f"Analysis parameters: {params}")

        # Check cache first
        cached_results = get_cached_results(directory, params)
        if cached_results:
            logger.info("Using cached results")
            analyzer = WallpaperAnalyzer()
            formatted_results = analyzer.format_results_for_frontend(cached_results)
            logger.info(f"Returning {len(formatted_results['images'])} images from cache")
            return jsonify({
                'success': True,
                'images': formatted_results['images']
            })

        # Perform analysis
        logger.info("Starting new analysis")
        analyzer = WallpaperAnalyzer()
        results = analyzer.analyze_directory(
            directory,
            similarity_threshold=params['similarity_threshold'],
            aesthetic_threshold=params['aesthetic_threshold'],
            recursive=params['recursive'],
            skip_duplicates=params['skip_duplicates'],
            skip_aesthetics=params['skip_aesthetics'],
            limit=params['limit']
        )
        logger.info("Analysis completed")

        # Cache results
        cache_results(directory, params, results)
        logger.info("Results cached")

        formatted_results = analyzer.format_results_for_frontend(results)
        logger.info(f"Returning {len(formatted_results['images'])} images")
        
        response = {
            'success': True,
            'images': formatted_results['images']
        }
        logger.info("Sending response to frontend")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/image')
def serve_image() -> 'flask.Response':
    try:
        image_path = request.args.get('path')
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(image_path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<path:directory>')
def get_results(directory: str) -> 'flask.Response':
    try:
        if directory in analysis_results:
            return jsonify(analysis_results[directory])
        return jsonify({'error': 'Results not found'}), 404
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect() -> None:
    logger.info(f"Client connected: {request.sid}")
    active_connections.add(request.sid)
    socketio.emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect(client_id: str) -> None:
    if request.sid in active_connections:
        active_connections.remove(request.sid)

@socketio.on_error()
def error_handler(e: Exception) -> None:
    logger.error(f"Socket.IO error: {str(e)}")
    socketio.emit('error', {'message': str(e)})

def main() -> None:
    """Main entry point for the application"""
    try:
        with app.app_context():
            print(f"Server running at http://localhost:{Config.PORT}")
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