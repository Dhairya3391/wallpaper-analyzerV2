import os
import logging
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from collections import defaultdict
import json
from datetime import datetime
from sklearn.cluster import DBSCAN
import hashlib
import base64
import platform
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm

# Mac-specific optimizations
if platform.system() == 'Darwin':
    # Enable Mac-specific optimizations
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # Use Metal Performance Shaders for better performance
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) available for Mac optimization")

# Configuration constants (functional approach - no classes)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("cpu")

BATCH_SIZE = 256 if torch.backends.mps.is_available() else \
             128 if torch.cuda.is_available() else \
             32  # Increased for modern CPUs

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_AESTHETIC_THRESHOLD = 0.8

DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 3

HOST = '0.0.0.0'
PORT = 8000
DEBUG = False

ENABLE_APP_LOGGING = False  # Disable verbose logging

CACHE_DIR = 'image_cache'
MAX_HASH_CACHE = 2000

# Minimal logging setup - only errors and warnings
handlers = [logging.FileHandler('analyzed.log')]
if ENABLE_APP_LOGGING:
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger('WallpaperAnalyzer')

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Global models (initialized once)
feature_model = None
aesthetic_model = None
transform = None

def initialize_models() -> None:
    """Initialize neural network models globally with Mac optimization"""
    global feature_model, aesthetic_model, transform
    
    print(f"🔄 Initializing models on device: {DEVICE}")
    
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
        feature_model = feature_model.to(DEVICE)
        feature_model.eval()
        
        # Aesthetic evaluation model - using ResNet50 for better accuracy
        aesthetic_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = aesthetic_model.fc.in_features
        aesthetic_model.fc = torch.nn.Linear(num_features, 1)
        aesthetic_model = aesthetic_model.to(DEVICE)
        aesthetic_model.eval()
        
        print("✅ Models initialized successfully")
        
    except Exception as e:
        logger.warning(f"Failed to load modern models, falling back to original models: {e}")
        # Fallback to original models
        feature_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Remove the classifier layer to get features
        feature_model.classifier = torch.nn.Identity()
        feature_model = feature_model.to(DEVICE)
        feature_model.eval()
        
        aesthetic_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = aesthetic_model.fc.in_features
        aesthetic_model.fc = torch.nn.Linear(num_features, 1)
        aesthetic_model = aesthetic_model.to(DEVICE)
        aesthetic_model.eval()

def get_image_hash(image_path: str) -> Optional[str]:
    """Calculate SHA256 hash of image file"""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {image_path}: {e}")
        return None

def get_cache_file_path(image_hash: str) -> str:
    """Get the cache file path for an image hash"""
    return os.path.join(CACHE_DIR, f"{image_hash}.json")

def load_cached_data(image_hash: str) -> Dict[str, Any]:
    """Load cached data for an image"""
    cache_file = get_cache_file_path(image_hash)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache for {image_hash}: {e}")
    return {}

def save_cached_data(image_hash: str, data: Dict[str, Any]) -> None:
    """Save data to cache for an image"""
    cache_file = get_cache_file_path(image_hash)
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving cache for {image_hash}: {e}")

def update_cached_data(image_hash: str, updates: Dict[str, Any]) -> None:
    """Update specific fields in cached data"""
    current_data = load_cached_data(image_hash)
    current_data.update(updates)
    current_data['last_updated'] = datetime.now().isoformat()
    save_cached_data(image_hash, current_data)

def cleanup_old_cache(max_age_days: int = 30) -> None:
    """Clean up old cache files to prevent disk space issues"""
    try:
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
        
        for cache_file in cache_files:
            cache_path = os.path.join(CACHE_DIR, cache_file)
            if os.path.getmtime(cache_path) < cutoff_time:
                os.remove(cache_path)
    except Exception as e:
        logger.warning(f"Error during cache cleanup: {e}")

def generate_analysis_cache_key(directory: str, recursive: bool, similarity_threshold: float, 
                               aesthetic_threshold: float) -> str:
    """Generate cache key for analysis parameters (excluding limit)"""
    # Create a stable cache key that doesn't include limit
    key_string = f"{directory}_{recursive}_{similarity_threshold}_{aesthetic_threshold}"
    return hashlib.sha256(key_string.encode()).hexdigest()

def get_analysis_cache_file(cache_key: str) -> str:
    """Get the cache file path for analysis results"""
    return os.path.join(CACHE_DIR, f"analysis_{cache_key}.json")

def load_analysis_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Load cached analysis results"""
    cache_file = get_analysis_cache_file(cache_key)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                return cached_data
        except Exception as e:
            logger.error(f"Error loading analysis cache: {e}")
    return None

def save_analysis_cache(cache_key: str, data: Dict[str, Any]) -> None:
    """Save analysis results to cache"""
    cache_file = get_analysis_cache_file(cache_key)
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving analysis cache: {e}")

def validate_cache_integrity(cached_data: Dict[str, Any], current_paths: List[str]) -> bool:
    """Validate if cached data is still valid for current directory state"""
    try:
        cached_paths = set(cached_data.get('image_paths', []))
        current_paths_set = set(current_paths)
        
        # Check if all current paths are in cached paths
        if not current_paths_set.issubset(cached_paths):
            return False
            
        # Check if cache is not too old (7 days)
        timestamp_str = cached_data.get('timestamp')
        if timestamp_str:
            try:
                cache_time = datetime.fromisoformat(timestamp_str)
                age_days = (datetime.now() - cache_time).days
                if age_days > 7:
                    return False
            except:
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error validating cache integrity: {e}")
        return False

def apply_limit_to_results(results: Dict[str, Any], limit: int) -> Dict[str, Any]:
    """Apply limit to results while preserving structure"""
    if limit <= 0:
        return results
        
    limited_results = results.copy()
    
    # Limit clusters
    if 'clusters' in results:
        cluster_items = list(results['clusters'].items())
        limited_clusters = {}
        
        total_images = 0
        for cluster_id, cluster_data in cluster_items:
            if total_images >= limit:
                break
                
            cluster_size = cluster_data['size']
            if total_images + cluster_size <= limit:
                limited_clusters[cluster_id] = cluster_data
                total_images += cluster_size
            else:
                # Partial cluster
                remaining = limit - total_images
                limited_paths = cluster_data['paths'][:remaining]
                limited_clusters[cluster_id] = {
                    'size': remaining,
                    'representative': limited_paths[0] if limited_paths else cluster_data['representative'],
                    'paths': limited_paths
                }
                total_images = limit
                break
                
        limited_results['clusters'] = limited_clusters
    
    # Limit duplicates
    if 'duplicates' in results:
        limited_duplicates = []
        total_duplicates = 0
        
        for group in results['duplicates']:
            if total_duplicates >= limit:
                break
                
            if total_duplicates + len(group) <= limit:
                limited_duplicates.append(group)
                total_duplicates += len(group)
            else:
                # Partial group
                remaining = limit - total_duplicates
                limited_duplicates.append(group[:remaining])
                total_duplicates = limit
                break
                
        limited_results['duplicates'] = limited_duplicates
    
    # Limit low aesthetic images
    if 'low_aesthetic' in results:
        limited_results['low_aesthetic'] = results['low_aesthetic'][:limit]
    
    # Update total count
    limited_results['total_images'] = min(results['total_images'], limit)
    
    return limited_results

def get_cache_statistics() -> Dict[str, Any]:
    """Get statistics about the cache"""
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
        analysis_cache_files = [f for f in cache_files if f.startswith('analysis_')]
        image_cache_files = [f for f in cache_files if not f.startswith('analysis_')]
        
        total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in cache_files)
        
        return {
            'total_cache_files': len(cache_files),
            'analysis_cache_files': len(analysis_cache_files),
            'image_cache_files': len(image_cache_files),
            'total_cache_size_mb': round(total_size / (1024 * 1024), 2)
        }
    except Exception as e:
        logger.error(f"Error getting cache statistics: {e}")
        return {}

def clear_cache(cache_type: str = 'all') -> Dict[str, Any]:
    """Clear cache files"""
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
        deleted_count = 0
        
        for cache_file in cache_files:
            should_delete = False
            
            if cache_type == 'all':
                should_delete = True
            elif cache_type == 'analysis' and cache_file.startswith('analysis_'):
                should_delete = True
            elif cache_type == 'images' and not cache_file.startswith('analysis_'):
                should_delete = True
                
            if should_delete:
                os.remove(os.path.join(CACHE_DIR, cache_file))
                deleted_count += 1
        
        return {
            'success': True,
            'deleted_files': deleted_count,
            'cache_type': cache_type
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return {'success': False, 'error': str(e)}

def get_cache_hit_rate() -> Dict[str, Any]:
    """Calculate cache hit rate (simplified)"""
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
        analysis_cache_files = [f for f in cache_files if f.startswith('analysis_')]
        
        return {
            'analysis_cache_count': len(analysis_cache_files),
            'image_cache_count': len(cache_files) - len(analysis_cache_files)
        }
    except Exception as e:
        logger.error(f"Error calculating cache hit rate: {e}")
        return {}

def calculate_perceptual_hash(image_path: str) -> Optional[bytes]:
    """Calculate perceptual hash of image"""
    try:
        with Image.open(image_path) as img:
            # Resize to 8x8 and convert to grayscale
            img = img.resize((8, 8), Image.Resampling.LANCZOS).convert('L')
            # Convert to numpy array
            pixels = np.array(img)
            # Calculate mean
            mean = pixels.mean()
            # Create hash
            hash_value = pixels > mean
            return hash_value.tobytes()
    except Exception as e:
        logger.error(f"Error calculating perceptual hash for {image_path}: {e}")
        return None

def process_single_image(image_path: str, process_type: str = 'features') -> Optional[Union[List[float], float]]:
    """Process a single image for features or aesthetic score"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                if process_type == 'features':
                    features = feature_model(img_tensor)
                    return features.squeeze().cpu().numpy().tolist()
                elif process_type == 'aesthetic':
                    score = aesthetic_model(img_tensor)
                    return torch.sigmoid(score).item()
                    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None

def process_images_sequential(image_paths: List[str], process_type: str = 'features') -> Dict[str, Union[List[float], float]]:
    """Process images sequentially with progress bar"""
    results = {}
    total = len(image_paths)
    
    if total == 0:
        return results
    
    # Create progress bar
    desc = f"Processing {process_type}"
    pbar = tqdm(image_paths, desc=desc, unit="img", ncols=80)
    
    for path in pbar:
        pbar.set_postfix({"file": os.path.basename(path)[:20]})
        result = process_single_image(path, process_type)
        if result is not None:
            results[path] = result
    
    pbar.close()
    return results

def find_duplicates_sequential(image_paths: List[str], threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> List[List[str]]:
    """Find duplicate images using sequential processing"""
    if not image_paths:
        return []

    print("🔍 Finding duplicates...")
    
    # Phase 1: Hash-based grouping
    image_hashes = defaultdict(list)
    pbar = tqdm(image_paths, desc="Calculating hashes", unit="img", ncols=80)
    for path in pbar:
        hash_value = calculate_perceptual_hash(path)
        if hash_value:
            image_hashes[hash_value].append(path)
    pbar.close()

    potential_duplicates = [paths for paths in image_hashes.values() if len(paths) > 1]
    
    # Phase 2: Feature-based comparison for remaining images
    if len(image_paths) <= 1000:  # Limit for performance
        features = process_images_sequential(image_paths, 'features')
        
        # Calculate similarity matrix
        paths = list(features.keys())
        if paths:
            print("🔍 Computing similarity matrix...")
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

def extract_features_sequential(image_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Extract features from images sequentially"""
    features = []
    valid_paths = []
    
    pbar = tqdm(image_paths, desc="Extracting features", unit="img", ncols=80)
    for path in pbar:
        try:
            feature = process_single_image(path, 'features')
            if feature is not None:
                features.append(feature)
                valid_paths.append(path)
        except Exception as e:
            logger.error(f"Error extracting features for {path}: {e}")
            continue
    pbar.close()
    
    return np.array(features), valid_paths

def cluster_images_dbscan(image_paths: List[str]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Cluster images using DBSCAN"""
    print("📊 Clustering images...")
    features, valid_paths = extract_features_sequential(image_paths)
    
    if len(features) < DBSCAN_MIN_SAMPLES:
        return {0: {"paths": valid_paths, "size": len(valid_paths)}}, {}
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
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
    
    # If no clusters found, create a default cluster with all images
    if not clusters:
        clusters[0] = {
            "size": len(valid_paths),
            "representative": valid_paths[0] if valid_paths else "",
            "paths": valid_paths
        }
    
    return clusters, {}

def get_image_paths(directory: str, recursive: bool = False) -> List[str]:
    """Get all image paths in a directory"""
    image_paths = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file.lower())[1] in IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if os.path.splitext(file.lower())[1] in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(directory, file))
    
    return image_paths

def analyze_directory(directory: str, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                     aesthetic_threshold: float = DEFAULT_AESTHETIC_THRESHOLD, recursive: bool = False,
                     skip_duplicates: bool = False, skip_aesthetics: bool = False, limit: int = 0) -> Dict[str, Any]:
    """Analyze a directory of images using functional approach with robust caching"""
    import time
    t0 = time.time()
    
    print(f"📁 Analyzing directory: {directory}")
    
    # 1. Load image paths (without limit first)
    all_image_paths = get_image_paths(directory, recursive)
    print(f"📸 Found {len(all_image_paths)} images")
    
    # 2. Generate cache key (excluding limit for stable caching)
    cache_key = generate_analysis_cache_key(directory, recursive, similarity_threshold, aesthetic_threshold)
    
    # 3. Try to load cached analysis results
    cached_data = load_analysis_cache(cache_key)
    if cached_data and validate_cache_integrity(cached_data, all_image_paths):
        print("✅ Using cached analysis results")
        cached_results = cached_data['results']
        
        # Apply limit to cached results if needed
        if limit > 0:
            results = apply_limit_to_results(cached_results, limit)
        else:
            results = cached_results
            
        print(f"✅ Analysis complete - {results['total_images']} images")
        return results
    
    # 4. Cache miss or invalid - need to perform analysis
    print("🔄 Performing new analysis...")
    
    # 5. Check individual image cache and process new images
    cached_features = {}
    cached_scores = {}
    to_process = []
    
    print("🔍 Checking image cache...")
    pbar = tqdm(all_image_paths, desc="Checking cache", unit="img", ncols=80)
    for path in pbar:
        image_hash = get_image_hash(path)
        if image_hash:
            cached_data = load_cached_data(image_hash)
            
            if 'features' in cached_data and 'aesthetic_score' in cached_data:
                cached_features[path] = cached_data['features']
                cached_scores[path] = cached_data['aesthetic_score']
            else:
                to_process.append((path, image_hash))
    pbar.close()
    
    # 6. Process new images
    features = {}
    scores = {}
    if to_process:
        print(f"🔄 Processing {len(to_process)} new images...")
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
    
    # 7. Duplicate detection
    duplicates = []
    if not skip_duplicates:
        duplicates = find_duplicates_sequential(all_image_paths, threshold=similarity_threshold)
    
    t2 = time.time()
    
    # 8. Clustering
    clusters = {}
    cluster_features = {}
    if not skip_aesthetics:
        cluster_result = cluster_images_dbscan(all_image_paths)
        if isinstance(cluster_result, tuple):
            clusters, cluster_features = cluster_result
        else:
            clusters = cluster_result
    
    t3 = time.time()
    
    # 9. Aesthetic scoring (already done above)
    aesthetic_scores = scores
    
    t4 = time.time()
    
    print(f"⏱️  Timing: loading: {t1-t0:.1f}s, duplicates: {t2-t1:.1f}s, clustering: {t3-t2:.1f}s, total: {t4-t0:.1f}s")
    
    # 10. Prepare full results (without limit)
    full_results = {
        'directory': directory,
        'total_images': len(all_image_paths),
        'processed_images': len(all_image_paths),
        'all_image_paths': all_image_paths,  # Include all image paths for processing
        'duplicates': duplicates,
        'aesthetic_scores': aesthetic_scores,
        'clusters': clusters,
        'cluster_features': cluster_features,
        'low_aesthetic': [path for path, score in aesthetic_scores.items() if score < aesthetic_threshold],
    }
    
    # 11. Save full analysis cache (without limit)
    try:
        analysis_cache_data = {
            'image_paths': all_image_paths,
            'analysis_params': {
                'similarity_threshold': similarity_threshold,
                'aesthetic_threshold': aesthetic_threshold,
                'recursive': recursive
            },
            'results': full_results,
            'timestamp': datetime.now().isoformat()
        }
        save_analysis_cache(cache_key, analysis_cache_data)
    except Exception as e:
        logger.warning(f"Failed to save analysis cache: {e}")
    
    # 12. Apply limit and return results
    if limit > 0:
        results = apply_limit_to_results(full_results, limit)
    else:
        results = full_results
    
    print(f"✅ Analysis completed - {results['total_images']} images")
    return results

def format_results_for_frontend(results: Dict[str, Any]) -> Dict[str, Any]:
    """Format analysis results for frontend display"""
    try:
        images = []
        clusters = []
        duplicates = []
        low_aesthetic = []
        
        # Create a set of all duplicate paths for quick lookup
        duplicate_paths = set()
        for group in results['duplicates']:
            duplicate_paths.update(group)
        
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
                    'is_duplicate': path in duplicate_paths,
                    'is_low_aesthetic': results['aesthetic_scores'].get(path, 0) < DEFAULT_AESTHETIC_THRESHOLD,
                    'aesthetic_score': results['aesthetic_scores'].get(path, 0)
                })
        
        # Process all images that are not in any cluster (noise points)
        clustered_paths = set()
        for cluster_data in results['clusters'].values():
            clustered_paths.update(cluster_data['paths'])
        
        # Add images that are not in any cluster
        for path in results.get('all_image_paths', []):
            if path not in clustered_paths:
                images.append({
                    'path': path,
                    'cluster': -1,  # -1 indicates no cluster (noise point)
                    'cluster_size': 1,
                    'is_duplicate': path in duplicate_paths,
                    'is_low_aesthetic': results['aesthetic_scores'].get(path, 0) < DEFAULT_AESTHETIC_THRESHOLD,
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
        
        params = {
            'similarity_threshold': float(data.get('similarity_threshold', DEFAULT_SIMILARITY_THRESHOLD)),
            'aesthetic_threshold': float(data.get('aesthetic_threshold', DEFAULT_AESTHETIC_THRESHOLD)),
            'recursive': bool(data.get('recursive', True)),
            'skip_duplicates': bool(data.get('skip_duplicates', False)),
            'skip_aesthetics': bool(data.get('skip_aesthetics', False)),
            'limit': int(data.get('limit', 0))
        }

        # Perform analysis
        results = analyze_directory(
            directory,
            similarity_threshold=params['similarity_threshold'],
            aesthetic_threshold=params['aesthetic_threshold'],
            recursive=params['recursive'],
            skip_duplicates=params['skip_duplicates'],
            skip_aesthetics=params['skip_aesthetics'],
            limit=params['limit']
        )

        formatted_results = format_results_for_frontend(results)
        
        response = {
            'success': True,
            'images': formatted_results['images']
        }
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

@app.route('/api/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics"""
    try:
        stats = get_cache_statistics()
        hit_rate = get_cache_hit_rate()
        stats.update(hit_rate)
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache_endpoint():
    """Clear cache files"""
    try:
        data = request.get_json() or {}
        cache_type = data.get('cache_type', 'all')
        
        result = clear_cache(cache_type)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cache/validate', methods=['POST'])
def validate_cache():
    """Validate cache integrity for a directory"""
    try:
        data = request.get_json()
        directory = data.get('directory')
        recursive = bool(data.get('recursive', True))
        similarity_threshold = float(data.get('similarity_threshold', DEFAULT_SIMILARITY_THRESHOLD))
        aesthetic_threshold = float(data.get('aesthetic_threshold', DEFAULT_AESTHETIC_THRESHOLD))
        
        # Get current image paths
        current_paths = get_image_paths(directory, recursive)
        
        # Generate cache key
        cache_key = generate_analysis_cache_key(directory, recursive, similarity_threshold, aesthetic_threshold)
        
        # Try to load cache
        cached_data = load_analysis_cache(cache_key)
        
        if cached_data:
            is_valid = validate_cache_integrity(cached_data, current_paths)
            return jsonify({
                'success': True,
                'cache_exists': True,
                'cache_valid': is_valid,
                'cached_images': len(cached_data.get('image_paths', [])),
                'current_images': len(current_paths)
            })
        else:
            return jsonify({
                'success': True,
                'cache_exists': False,
                'cache_valid': False,
                'current_images': len(current_paths)
            })
            
    except Exception as e:
        logger.error(f"Error validating cache: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def main():
    """Main entry point for the application"""
    try:
        print(f"🚀 Wallpaper Analyzer V2 starting...")
        print(f"🌐 Server running at http://localhost:{PORT}")
        print("✅ Caching system enabled")
        app.run(
            host=HOST,
            port=PORT,
            debug=DEBUG,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")

if __name__ == '__main__':
    main() 