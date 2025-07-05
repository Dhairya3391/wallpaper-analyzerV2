# Wallpaper Analyzer V2
# Functional approach with JSON file system for per-picture data storage
# Single-threaded processing with DBSCAN clustering
# Mac-optimized with latest dependencies

import os
import logging
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from functools import lru_cache
from collections import defaultdict
import json
from datetime import datetime
from sklearn.cluster import DBSCAN
import hashlib
import pickle
import base64
import platform

# Mac-specific optimizations
if platform.system() == 'Darwin':
    # Enable Mac-specific optimizations
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # Use Metal Performance Shaders for better performance
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) available for Mac optimization")

# Configuration
class Config:
    # Device selection with Mac optimization
    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else \
             torch.device("cuda") if torch.cuda.is_available() else \
             torch.device("cpu")
    
    # Optimized batch sizes for different devices
    BATCH_SIZE = 256 if torch.backends.mps.is_available() else \
                 128 if torch.cuda.is_available() else \
                 32  # Increased for modern CPUs
    
    # Image formats supported
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    DEFAULT_SIMILARITY_THRESHOLD = 0.85
    DEFAULT_AESTHETIC_THRESHOLD = 0.8
    
    # Clustering settings
    DBSCAN_EPS = 0.3
    DBSCAN_MIN_SAMPLES = 3
    
    # Flask settings
    HOST = '0.0.0.0'
    PORT = 8000
    DEBUG = False
    
    # Logging settings
    ENABLE_APP_LOGGING = True
    
    # JSON storage settings
    CACHE_DIR = 'image_cache'
    MAX_HASH_CACHE = 2000
    
    # Mac-specific settings
    USE_MPS_FALLBACK = True if platform.system() == 'Darwin' else False

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

# Ensure cache directory exists
os.makedirs(Config.CACHE_DIR, exist_ok=True)

# Global models (initialized once)
feature_model = None
aesthetic_model = None
transform = None

def initialize_models():
    """Initialize neural network models globally with Mac optimization"""
    global feature_model, aesthetic_model, transform
    
    logger.info(f"Initializing models on device: {Config.DEVICE}")
    
    # Enhanced transform with better image quality
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Use more modern and efficient models
        # Feature extraction model - using EfficientNet for better performance
        feature_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Remove the classifier layer to get features
        feature_model.classifier = torch.nn.Identity()
        feature_model = feature_model.to(Config.DEVICE)
        feature_model.eval()
        
        # Aesthetic evaluation model - using ResNet50 for better accuracy
        aesthetic_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = aesthetic_model.fc.in_features
        aesthetic_model.fc = torch.nn.Linear(num_features, 1)
        aesthetic_model = aesthetic_model.to(Config.DEVICE)
        aesthetic_model.eval()
        
        logger.info("✅ Models initialized successfully with modern architectures")
        
    except Exception as e:
        logger.warning(f"Failed to load modern models, falling back to original models: {e}")
        # Fallback to original models
        feature_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Remove the classifier layer to get features
        feature_model.classifier = torch.nn.Identity()
        feature_model = feature_model.to(Config.DEVICE)
        feature_model.eval()
        
        aesthetic_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = aesthetic_model.fc.in_features
        aesthetic_model.fc = torch.nn.Linear(num_features, 1)
        aesthetic_model = aesthetic_model.to(Config.DEVICE)
        aesthetic_model.eval()

def get_image_hash(image_path: str) -> str:
    """Calculate SHA256 hash of image file"""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {image_path}: {e}")
        return None

def get_cache_file_path(image_hash: str) -> str:
    """Get the cache file path for an image hash"""
    return os.path.join(Config.CACHE_DIR, f"{image_hash}.json")

def load_cached_data(image_hash: str) -> dict:
    """Load cached data for an image"""
    cache_file = get_cache_file_path(image_hash)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache for {image_hash}: {e}")
    return {}

def save_cached_data(image_hash: str, data: dict):
    """Save data to cache for an image"""
    cache_file = get_cache_file_path(image_hash)
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving cache for {image_hash}: {e}")

def update_cached_data(image_hash: str, updates: dict):
    """Update specific fields in cached data"""
    current_data = load_cached_data(image_hash)
    current_data.update(updates)
    current_data['last_updated'] = datetime.now().isoformat()
    save_cached_data(image_hash, current_data)

def cleanup_old_cache(max_age_days: int = 30):
    """Clean up old cache files to prevent disk space issues"""
    try:
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        cache_files = [f for f in os.listdir(Config.CACHE_DIR) if f.endswith('.json')]
        
        for cache_file in cache_files:
            cache_path = os.path.join(Config.CACHE_DIR, cache_file)
            if os.path.getmtime(cache_path) < cutoff_time:
                os.remove(cache_path)
                logger.info(f"Cleaned up old cache file: {cache_file}")
    except Exception as e:
        logger.warning(f"Error during cache cleanup: {e}")

def calculate_perceptual_hash(image_path: str) -> bytes:
    """Calculate perceptual hash of an image"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
        with torch.no_grad():
            features = feature_model(image_tensor)
            # Handle different output shapes based on model type
            if features.dim() == 4:  # (batch, channels, height, width)
                features = torch.mean(features, dim=[2, 3])  # Global average pooling
            elif features.dim() == 2:  # (batch, features)
                features = features.squeeze(0)
            else:
                features = features.flatten()
            return features.cpu().numpy().tobytes()
    except Exception as e:
        logger.error(f"Error calculating perceptual hash for {image_path}: {e}")
        return None

def process_single_image(image_path: str, process_type: str = 'features'):
    """Process a single image for features or aesthetic scoring"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
        
        with torch.no_grad():
            if process_type == 'features':
                outputs = feature_model(image_tensor)
                # Handle different output shapes based on model type
                if outputs.dim() == 4:  # (batch, channels, height, width)
                    outputs = torch.mean(outputs, dim=[2, 3])  # Global average pooling
                elif outputs.dim() == 2:  # (batch, features)
                    outputs = outputs.squeeze(0)
                else:
                    outputs = outputs.flatten()
            else:  # aesthetic scoring
                outputs = aesthetic_model(image_tensor)
                outputs = torch.sigmoid(outputs).squeeze()
            
            result = outputs.cpu().numpy()
            if result.ndim == 0:
                result = float(result)
            else:
                result = result.tolist()
            
            return result
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def process_images_sequential(image_paths: list[str], process_type: str = 'features') -> dict:
    """Process images sequentially (no multithreading)"""
    results = {}
    total = len(image_paths)
    
    for i, path in enumerate(image_paths):
        logger.info(f"Processing {process_type} for image {i+1}/{total}: {os.path.basename(path)}")
        result = process_single_image(path, process_type)
        if result is not None:
            results[path] = result
    
    logger.info(f"Completed {process_type} for {len(results)} images.")
    return results

def find_duplicates_sequential(image_paths: list[str], threshold: float = Config.DEFAULT_SIMILARITY_THRESHOLD) -> list[list[str]]:
    """Find duplicate images using sequential processing"""
    if not image_paths:
        return []

    # Phase 1: Hash-based grouping
    image_hashes = defaultdict(list)
    for path in image_paths:
        hash_value = calculate_perceptual_hash(path)
        if hash_value:
            image_hashes[hash_value].append(path)

    potential_duplicates = [paths for paths in image_hashes.values() if len(paths) > 1]
    
    # Phase 2: Feature-based comparison for remaining images
    if len(image_paths) <= 1000:  # Limit for performance
        features = process_images_sequential(image_paths, 'features')
        
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

def extract_features_sequential(image_paths: list[str]) -> tuple[np.ndarray, list[str]]:
    """Extract features from images sequentially"""
    features = []
    valid_paths = []
    
    for path in image_paths:
        try:
            feature = process_single_image(path, 'features')
            if feature is not None:
                features.append(feature)
                valid_paths.append(path)
        except Exception as e:
            logger.error(f"Error extracting features for {path}: {e}")
            continue
    
    return np.array(features), valid_paths

def cluster_images_dbscan(image_paths: list[str]) -> tuple[dict, dict]:
    """Cluster images using DBSCAN"""
    features, valid_paths = extract_features_sequential(image_paths)
    
    if len(features) < Config.DBSCAN_MIN_SAMPLES:
        return {0: {"paths": valid_paths, "size": len(valid_paths)}}, {}
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=Config.DBSCAN_EPS, min_samples=Config.DBSCAN_MIN_SAMPLES)
    cluster_labels = dbscan.fit_predict(features)
    
    # Create cluster information
    clusters = {}
    unique_labels = np.unique(cluster_labels)
    
    for label in unique_labels:
        if label == -1:  # Noise points
            continue
            
        cluster_indices = np.where(cluster_labels == label)[0]
        cluster_paths = [valid_paths[idx] for idx in cluster_indices]
        cluster_features = features[cluster_indices]
        
        # Find representative image (closest to cluster center)
        center = np.mean(cluster_features, axis=0)
        distances = np.linalg.norm(cluster_features - center, axis=1)
        representative_idx = np.argmin(distances)
        
        clusters[label] = {
            "size": len(cluster_paths),
            "representative": cluster_paths[representative_idx],
            "paths": cluster_paths
        }
    
    # Handle noise points as individual clusters
    noise_indices = np.where(cluster_labels == -1)[0]
    for i, idx in enumerate(noise_indices):
        noise_label = f"noise_{i}"
        clusters[noise_label] = {
            "size": 1,
            "representative": valid_paths[idx],
            "paths": [valid_paths[idx]]
        }
    
    return clusters, {}

def get_image_paths(directory: str, recursive: bool = False) -> list[str]:
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

def analyze_directory(directory: str, similarity_threshold: float = Config.DEFAULT_SIMILARITY_THRESHOLD,
                     aesthetic_threshold: float = Config.DEFAULT_AESTHETIC_THRESHOLD, recursive: bool = False,
                     skip_duplicates: bool = False, skip_aesthetics: bool = False, limit: int = 0) -> dict:
    """Analyze a directory of images using functional approach"""
    import time
    t0 = time.time()
    
    logger.info(f"Starting analysis for {directory}")
    
    # 1. Load image paths
    logger.info("Loading image paths...")
    image_paths = get_image_paths(directory, recursive)
    if limit and limit > 0:
        image_paths = image_paths[:limit]
    logger.info(f"Loaded {len(image_paths)} image paths.")
    
    # Check for directory-level cache
    directory_hash = hashlib.sha256(f"{directory}_{recursive}_{limit}_{similarity_threshold}_{aesthetic_threshold}".encode()).hexdigest()
    directory_cache_file = os.path.join(Config.CACHE_DIR, f"dir_{directory_hash}.json")
    
    if os.path.exists(directory_cache_file):
        try:
            with open(directory_cache_file, 'r') as f:
                cached_analysis = json.load(f)
            
            # Check if all images in the cached analysis still exist
            cached_paths = set(cached_analysis.get('image_paths', []))
            current_paths = set(image_paths)
            
            if cached_paths == current_paths:
                logger.info("✅ Using cached analysis results")
                logger.info(f"Cache contains {len(cached_analysis['results'].get('duplicates', []))} duplicate groups and {len(cached_analysis['results'].get('clusters', {}))} clusters")
                return cached_analysis['results']
            else:
                logger.info(f"Cache outdated - {len(cached_paths - current_paths)} images removed, {len(current_paths - cached_paths)} images added")
        except Exception as e:
            logger.warning(f"Error reading directory cache: {e}")
    
    # 2. Check cache and process new images
    cached_features = {}
    cached_scores = {}
    to_process = []
    
    for path in image_paths:
        image_hash = get_image_hash(path)
        if image_hash:
            cached_data = load_cached_data(image_hash)
            
            if 'features' in cached_data and 'aesthetic_score' in cached_data:
                cached_features[path] = cached_data['features']
                cached_scores[path] = cached_data['aesthetic_score']
            else:
                to_process.append((path, image_hash))
    
    # 3. Process new images
    features = {}
    scores = {}
    if to_process:
        logger.info(f"Processing {len(to_process)} new images...")
        paths = [p[0] for p in to_process]
        
        # Feature extraction
        new_features = process_images_sequential(paths, 'features')
        # Aesthetic scoring
        new_scores = process_images_sequential(paths, 'aesthetic')
        
        # Store in cache
        for (path, image_hash) in to_process:
            feat = new_features.get(path)
            score = new_scores.get(path)
            if feat is not None:
                features[path] = feat
            if score is not None:
                scores[path] = score
            
            # Save to cache
            cache_data = {
                'path': path,
                'features': feat,
                'aesthetic_score': score,
                'last_updated': datetime.now().isoformat()
            }
            save_cached_data(image_hash, cache_data)
    
    # Merge cached and new
    features.update(cached_features)
    scores.update(cached_scores)
    
    t1 = time.time()
    
    # 4. Duplicate detection
    logger.info("Starting duplicate detection...")
    duplicates = []
    if not skip_duplicates:
        duplicates = find_duplicates_sequential(image_paths, threshold=similarity_threshold)
    logger.info("Duplicate detection complete.")
    
    t2 = time.time()
    
    # 5. Clustering
    logger.info("Starting clustering...")
    clusters = {}
    cluster_features = {}
    if not skip_aesthetics:
        cluster_result = cluster_images_dbscan(image_paths)
        if isinstance(cluster_result, tuple):
            clusters, cluster_features = cluster_result
        else:
            clusters = cluster_result
    logger.info("Clustering complete.")
    
    t3 = time.time()
    
    # 6. Aesthetic scoring (already done above)
    aesthetic_scores = scores
    logger.info("Aesthetic scoring complete.")
    
    t4 = time.time()
    
    logger.info(f"Timing: image loading: {t1-t0:.2f}s, duplicate detection: {t2-t1:.2f}s, clustering: {t3-t2:.2f}s, total: {t4-t0:.2f}s")
    
    # Prepare results
    results = {
        'directory': directory,
        'total_images': len(image_paths),
        'processed_images': len(image_paths),
        'duplicates': duplicates,
        'aesthetic_scores': aesthetic_scores,
        'clusters': clusters,
        'cluster_features': cluster_features,
        'low_aesthetic': [path for path, score in aesthetic_scores.items() if score < aesthetic_threshold],
    }
    
    # Save directory-level cache
    try:
        directory_cache_data = {
            'image_paths': image_paths,
            'analysis_params': {
                'similarity_threshold': similarity_threshold,
                'aesthetic_threshold': aesthetic_threshold,
                'recursive': recursive,
                'limit': limit
            },
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        with open(directory_cache_file, 'w') as f:
            json.dump(directory_cache_data, f, indent=2)
        logger.info(f"✅ Saved directory analysis cache: {directory_cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save directory cache: {e}")
    
    return results

def format_results_for_frontend(results: dict) -> dict:
    """Format analysis results for frontend display"""
    try:
        images = []
        clusters = []
        duplicates = []
        low_aesthetic = []
        
        # Process clusters
        for cluster_id, cluster_data in results['clusters'].items():
            clusters.append({
                'id': cluster_id,
                'size': cluster_data['size'],
                'representative': cluster_data['representative'],
                'paths': cluster_data['paths']
            })
            
            # Add images from this cluster
            for path in cluster_data['paths']:
                images.append({
                    'path': path,
                    'cluster': cluster_id,
                    'cluster_size': cluster_data['size'],
                    'is_duplicate': any(path in group for group in results['duplicates']),
                    'is_low_aesthetic': results['aesthetic_scores'].get(path, 0) < Config.DEFAULT_AESTHETIC_THRESHOLD,
                    'aesthetic_score': results['aesthetic_scores'].get(path, 0)
                })
        
        # Process duplicates
        for group in results['duplicates']:
            duplicates.extend(group)
        
        # Process low aesthetic images
        low_aesthetic = results['low_aesthetic']
        
        return {
            'images': images,
            'clusters': clusters,
            'duplicates': duplicates,
            'low_aesthetic': low_aesthetic
        }
    except Exception as e:
        logger.error(f"Error formatting results: {str(e)}")
        return {
            'images': [],
            'clusters': [],
            'duplicates': [],
            'low_aesthetic': []
        }

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Initialize models on startup
initialize_models()

# Clean up old cache files
cleanup_old_cache()

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        directory = data.get('directory')
        logger.info(f"Starting analysis for directory: {directory}")
        
        params = {
            'similarity_threshold': float(data.get('similarity_threshold', Config.DEFAULT_SIMILARITY_THRESHOLD)),
            'aesthetic_threshold': float(data.get('aesthetic_threshold', Config.DEFAULT_AESTHETIC_THRESHOLD)),
            'recursive': bool(data.get('recursive', True)),
            'skip_duplicates': bool(data.get('skip_duplicates', False)),
            'skip_aesthetics': bool(data.get('skip_aesthetics', False)),
            'limit': int(data.get('limit', 0))
        }
        logger.info(f"Analysis parameters: {params}")

        # Perform analysis
        logger.info("Starting new analysis")
        results = analyze_directory(
            directory,
            similarity_threshold=params['similarity_threshold'],
            aesthetic_threshold=params['aesthetic_threshold'],
            recursive=params['recursive'],
            skip_duplicates=params['skip_duplicates'],
            skip_aesthetics=params['skip_aesthetics'],
            limit=params['limit']
        )
        logger.info("Analysis completed")

        formatted_results = format_results_for_frontend(results)
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
def serve_image():
    try:
        image_path = request.args.get('path')
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(image_path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_aesthetic', methods=['POST'])
def update_aesthetic():
    """Update aesthetic score for an image"""
    try:
        data = request.get_json()
        image_path = data.get('path')
        new_score = float(data.get('aesthetic_score'))
        
        image_hash = get_image_hash(image_path)
        if image_hash:
            update_cached_data(image_hash, {'aesthetic_score': new_score})
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Could not calculate image hash'}), 400
            
    except Exception as e:
        logger.error(f"Error updating aesthetic score: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/update_duplicate', methods=['POST'])
def update_duplicate():
    """Update duplicate status for an image"""
    try:
        data = request.get_json()
        image_path = data.get('path')
        is_duplicate = bool(data.get('is_duplicate'))
        
        image_hash = get_image_hash(image_path)
        if image_hash:
            update_cached_data(image_hash, {'is_duplicate': is_duplicate})
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Could not calculate image hash'}), 400
            
    except Exception as e:
        logger.error(f"Error updating duplicate status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def main():
    """Main entry point for the application"""
    try:
        print(f"Server running at http://localhost:{Config.PORT}")
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")

if __name__ == '__main__':
    main() 