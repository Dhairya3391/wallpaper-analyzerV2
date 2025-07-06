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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hashlib
import platform
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
import concurrent.futures
import threading

# Mac-specific optimizations
if platform.system() == 'Darwin':
    # Enable Mac-specific optimizations
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # Use Metal Performance Shaders for better performance
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) available for Mac optimization")

# Configuration constants
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("cpu")

# Optimized batch sizes and thread counts
MAX_WORKERS = min(8, (os.cpu_count() or 1) + 4)  # Optimal for I/O bound tasks
BATCH_SIZE = 256 if torch.backends.mps.is_available() else \
             128 if torch.cuda.is_available() else \
             64

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_AESTHETIC_THRESHOLD = 0.85

# KMeans parameters
KMEANS_MAX_ITER = 300
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10

HOST = '0.0.0.0'
PORT = 8000
DEBUG = False

ENABLE_APP_LOGGING = False  # Disable verbose logging

CACHE_DIR = 'image_cache'

# Thread-safe logging setup
handlers = [logging.FileHandler('analyzed.log')]
if ENABLE_APP_LOGGING:
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger('Wallyzer')

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Global models and thread safety
feature_model = None
aesthetic_model = None
transform = None
model_lock = threading.Lock()

def initialize_models() -> None:
    """Initialize neural network models globally with thread safety"""
    global feature_model, aesthetic_model, transform
    
    with model_lock:
        if feature_model is not None:
            return  # Already initialized
        
        print(f"🔄 Initializing models on device: {DEVICE}")
        
        # Enhanced transform with better image quality
        transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Progress bar for model initialization
        with tqdm(total=2, desc="Loading models", unit="model", ncols=80) as pbar:
            try:
                # Use EfficientNet for better performance
                pbar.set_description("Loading feature model (EfficientNet)")
                feature_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                feature_model.classifier = torch.nn.Identity()
                feature_model = feature_model.to(DEVICE)
                feature_model.eval()
                pbar.update(1)
                
                # Aesthetic evaluation model
                pbar.set_description("Loading aesthetic model (ResNet50)")
                aesthetic_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                num_features = aesthetic_model.fc.in_features
                aesthetic_model.fc = torch.nn.Linear(num_features, 1)
                aesthetic_model = aesthetic_model.to(DEVICE)
                aesthetic_model.eval()
                pbar.update(1)
                
                print("✅ Models initialized successfully")
                
            except Exception as e:
                logger.warning(f"Failed to load modern models, falling back: {e}")
                # Fallback to lighter models
                pbar.set_description("Loading fallback feature model (MobileNet)")
                feature_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                feature_model.classifier = torch.nn.Identity()
                feature_model = feature_model.to(DEVICE)
                feature_model.eval()
                pbar.update(1)
                
                pbar.set_description("Loading fallback aesthetic model (ResNet18)")
                aesthetic_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                num_features = aesthetic_model.fc.in_features
                aesthetic_model.fc = torch.nn.Linear(num_features, 1)
                aesthetic_model = aesthetic_model.to(DEVICE)
                aesthetic_model.eval()
                pbar.update(1)

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
        data = safe_json_load(cache_file)
        return data if data is not None else {}
    return {}

def save_cached_data(image_hash: str, data: Dict[str, Any]) -> None:
    """Save data to cache for an image"""
    cache_file = get_cache_file_path(image_hash)
    if not safe_json_dump(data, cache_file):
        logger.warning(f"Failed to save cache for {image_hash}")

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
        
        if not cache_files:
            print("✅ No cache files to clean up")
            return
            
        print(f"🧹 Cleaning up cache files older than {max_age_days} days...")
        removed_count = 0
        
        with tqdm(cache_files, desc="Cleaning cache", unit="file", ncols=80) as pbar:
            for cache_file in pbar:
                cache_path = os.path.join(CACHE_DIR, cache_file)
                if os.path.getmtime(cache_path) < cutoff_time:
                    os.remove(cache_path)
                    removed_count += 1
                    pbar.set_postfix({"removed": removed_count})
        
        print(f"✅ Cache cleanup complete - removed {removed_count} files")
    except Exception as e:
        logger.warning(f"Error during cache cleanup: {e}")

def generate_analysis_cache_key(directory: str, recursive: bool, similarity_threshold: float, 
                               aesthetic_threshold: float, n_clusters: Optional[int] = None) -> str:
    """Generate cache key for analysis parameters (excluding limit)"""
    # Create a stable cache key that doesn't include limit
    key_string = f"{directory}_{recursive}_{similarity_threshold}_{aesthetic_threshold}"
    if n_clusters is not None:
        key_string += f"_{n_clusters}"
    return hashlib.sha256(key_string.encode()).hexdigest()

def get_analysis_cache_file(cache_key: str) -> str:
    """Get the cache file path for analysis results"""
    return os.path.join(CACHE_DIR, f"analysis_{cache_key}.json")

def load_analysis_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Load cached analysis results"""
    cache_file = get_analysis_cache_file(cache_key)
    if os.path.exists(cache_file):
        return safe_json_load(cache_file)
    return None

def safe_convert_numpy(obj):
    """Safely convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: safe_convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [safe_convert_numpy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(safe_convert_numpy(item) for item in obj)
    else:
        return obj

def safe_json_dump(data: Any, file_path: str) -> bool:
    """Safely dump data to JSON file with proper error handling"""
    try:
        # Convert numpy types before saving
        json_safe_data = safe_convert_numpy(data)
        
        # Write to temporary file first, then rename (atomic operation)
        temp_path = file_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(json_safe_data, f, indent=2, default=str)
        
        # Atomic rename
        os.rename(temp_path, file_path)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False

def safe_json_load(file_path: str) -> Optional[Dict[str, Any]]:
    """Safely load JSON file with error handling"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None

def save_analysis_cache(cache_key: str, data: Dict[str, Any]) -> None:
    """Save analysis results to cache"""
    cache_file = get_analysis_cache_file(cache_key)
    if not safe_json_dump(data, cache_file):
        logger.warning(f"Failed to save analysis cache for {cache_key}")

def validate_cache_integrity(cached_data: Dict[str, Any], current_paths: List[str], n_clusters: Optional[int] = None) -> bool:
    """Validate if cached data is still valid for current directory state"""
    try:
        cached_paths = set(cached_data.get('image_paths', []))
        current_paths_set = set(current_paths)
        
        # Check if all current paths are in cached paths
        if not current_paths_set.issubset(cached_paths):
            return False
            
        # Check if n_clusters parameter matches
        cached_params = cached_data.get('analysis_params', {})
        cached_n_clusters = cached_params.get('n_clusters')
        if cached_n_clusters != n_clusters:
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
        
        if not cache_files:
            print("✅ No cache files to clear")
            return {
                'success': True,
                'deleted_files': 0,
                'cache_type': cache_type
            }
        
        # Filter files based on cache type
        files_to_delete = []
        for cache_file in cache_files:
            should_delete = False
            
            if cache_type == 'all':
                should_delete = True
            elif cache_type == 'analysis' and cache_file.startswith('analysis_'):
                should_delete = True
            elif cache_type == 'images' and not cache_file.startswith('analysis_'):
                should_delete = True
                
            if should_delete:
                files_to_delete.append(cache_file)
        
        if not files_to_delete:
            print(f"✅ No {cache_type} cache files to clear")
            return {
                'success': True,
                'deleted_files': 0,
                'cache_type': cache_type
            }
        
        print(f"🗑️ Clearing {cache_type} cache ({len(files_to_delete)} files)...")
        deleted_count = 0
        
        with tqdm(files_to_delete, desc="Clearing cache", unit="file", ncols=80) as pbar:
            for cache_file in pbar:
                os.remove(os.path.join(CACHE_DIR, cache_file))
                deleted_count += 1
                pbar.set_postfix({"deleted": deleted_count})
        
        print(f"✅ Cache cleared - deleted {deleted_count} files")
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
    """Process a single image for features or aesthetic score with thread safety"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                if process_type == 'features':
                    with model_lock:
                        features = feature_model(img_tensor)
                    return features.squeeze().cpu().numpy().tolist()
                elif process_type == 'aesthetic':
                    with model_lock:
                        score = aesthetic_model(img_tensor)
                    return float(torch.sigmoid(score).item())
                    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None

def process_batch_images(image_paths: List[str], process_type: str = 'features') -> Dict[str, Union[List[float], float]]:
    """Process a batch of images with proper error handling"""
    results = {}
    
    for path in image_paths:
        try:
            result = process_single_image(path, process_type)
            if result is not None:
                results[path] = result
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            continue
    
    return results

def process_images_multithreaded(image_paths: List[str], process_type: str = 'features') -> Dict[str, Union[List[float], float]]:
    """Process images using multithreading with proper batch management"""
    if not image_paths:
        return {}
    
    # Create batches to avoid overwhelming the GPU
    batch_size = min(BATCH_SIZE, len(image_paths))
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    
    results = {}
    desc = f"Processing {process_type}"
    
    # Use ThreadPoolExecutor for I/O bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(process_batch_images, batch, process_type): batch 
            for batch in batches
        }
        
        # Process results with progress bar
        with tqdm(total=len(image_paths), desc=desc, unit="img", ncols=80) as pbar:
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.update(batch_results)
                    pbar.update(len(batch_results))
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    continue
    
    return results

def find_duplicates_multithreaded(image_paths: List[str], threshold: float = DEFAULT_SIMILARITY_THRESHOLD, 
                                 pre_computed_features: Optional[Dict[str, List[float]]] = None) -> List[List[str]]:
    """Find duplicate images using multithreading with optional pre-computed features"""
    if not image_paths:
        return []

    print("🔍 Finding duplicates...")
    
    # Phase 1: Hash-based grouping with multithreading
    def calculate_hash_batch(paths_batch):
        batch_hashes = {}
        for path in paths_batch:
            hash_value = calculate_perceptual_hash(path)
            if hash_value:
                batch_hashes[path] = hash_value
        return batch_hashes
    
    # Create batches for hash calculation
    batch_size = max(1, len(image_paths) // MAX_WORKERS)
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    
    # Calculate hashes in parallel
    all_hashes = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=len(image_paths), desc="Calculating hashes", unit="img", ncols=80) as pbar:
            futures = [executor.submit(calculate_hash_batch, batch) for batch in batches]
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_hashes = future.result()
                    all_hashes.update(batch_hashes)
                    pbar.update(len(batch_hashes))
                except Exception as e:
                    logger.error(f"Error in hash calculation batch: {e}")
    
    # Group by hash
    image_hashes = defaultdict(list)
    for path, hash_value in all_hashes.items():
        image_hashes[hash_value].append(path)
    
    potential_duplicates = [paths for paths in image_hashes.values() if len(paths) > 1]
    
    # Phase 2: Feature-based comparison for remaining images (if manageable)
    if len(image_paths) <= 1000:
        print("🔍 Computing similarity matrix...")
        
        # Use pre-computed features if available, otherwise compute them
        if pre_computed_features is not None:
            features = pre_computed_features
        else:
            features = process_images_multithreaded(image_paths, 'features')
        
        paths = list(features.keys())
        if paths:
            feature_matrix = np.stack([features[path] for path in paths])
            feature_matrix = feature_matrix / np.linalg.norm(feature_matrix, axis=1, keepdims=True)
            similarity_matrix = np.dot(feature_matrix, feature_matrix.T)
            
            # Find similar images
            processed = set()
            print("🔍 Finding similar images...")
            with tqdm(range(len(paths)), desc="Finding duplicates", unit="img", ncols=80) as pbar:
                for i in pbar:
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
                    pbar.set_postfix({"groups": len(potential_duplicates)})

    return potential_duplicates



def find_optimal_k_elbow(features: np.ndarray, max_k: int = 10) -> int:
    """Find optimal number of clusters using elbow method"""
    if len(features) < 2:
        return 1
    
    # Limit max_k to number of samples
    max_k = min(max_k, len(features) - 1)
    if max_k < 2:
        return 1
    
    inertias = []
    silhouette_scores = []
    k_values = range(1, max_k + 1)
    
    print(f"🔍 Finding optimal k using elbow method (testing k=1 to {max_k})...")
    
    with tqdm(k_values, desc="Testing k values", unit="k", ncols=80) as pbar:
        for k in pbar:
            pbar.set_postfix({"current_k": k})
            if k == 1:
                # For k=1, inertia is sum of squared distances to mean
                inertia = np.sum((features - np.mean(features, axis=0))**2)
                inertias.append(inertia)
                silhouette_scores.append(0)  # Silhouette score is not defined for k=1
            else:
                # Use KMeans for accuracy
                kmeans = KMeans(
                    n_clusters=k,
                    max_iter=KMEANS_MAX_ITER,
                    random_state=KMEANS_RANDOM_STATE,
                    n_init=KMEANS_N_INIT
                )
                cluster_labels = kmeans.fit_predict(features)
                inertias.append(kmeans.inertia_)
                
                # Calculate silhouette score
                try:
                    silhouette_avg = silhouette_score(features, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                except:
                    silhouette_scores.append(0)
    
    # Find elbow point using second derivative method
    if len(inertias) > 2:
        # Calculate second derivative of inertia
        second_derivative = np.diff(np.diff(inertias))
        # Find the point with maximum second derivative (elbow)
        elbow_idx = np.argmax(second_derivative) + 2  # +2 because we lost 2 points in diff
        optimal_k = k_values[elbow_idx]
    else:
        optimal_k = 1
    
    # Also consider silhouette score as a secondary criterion
    if len(silhouette_scores) > 1:
        best_silhouette_idx = np.argmax(silhouette_scores[1:]) + 1  # Skip k=1
        best_silhouette_k = k_values[best_silhouette_idx]
        
        # If silhouette suggests a different k, use the average
        if abs(optimal_k - best_silhouette_k) <= 2:
            optimal_k = (optimal_k + best_silhouette_k) // 2
    
    print(f"✅ Optimal k determined: {optimal_k}")
    return optimal_k

def cluster_images_kmeans(image_paths: List[str], n_clusters: Optional[int] = None) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Cluster images using KMeans with multithreading"""
    print("📊 Clustering images using KMeans...")
    
    # Use process_images_multithreaded directly instead of the wrapper
    features_dict = process_images_multithreaded(image_paths, 'features')
    
    if not features_dict:
        return {}, {}
    
    valid_paths = list(features_dict.keys())
    features = np.array([features_dict[path] for path in valid_paths])
    
    if len(features) == 0:
        return {}, {}
    
    if len(features) == 1:
        return {0: {"paths": valid_paths, "size": 1, "representative": valid_paths[0]}}, {}
    
    # Determine optimal number of clusters
    if n_clusters is None:
        n_clusters = find_optimal_k_elbow(features)
    
    n_clusters = max(1, min(n_clusters, len(features)))
    
    print(f"🎯 Using {n_clusters} clusters for {len(features)} images")
    
    # Apply clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iter=KMEANS_MAX_ITER,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT
    )
    cluster_labels = kmeans.fit_predict(features)
    
    # Create cluster information
    clusters = {}
    unique_labels = np.unique(cluster_labels)
    
    print("🔧 Processing clusters...")
    with tqdm(unique_labels, desc="Processing clusters", unit="cluster", ncols=80) as pbar:
        for label in pbar:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_paths = [valid_paths[idx] for idx in cluster_indices]
            cluster_features = features[cluster_indices]
            
            # Find representative image
            cluster_center = kmeans.cluster_centers_[label]
            distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
            representative_idx = np.argmin(distances)
            
            clusters[int(label)] = {
                "size": int(len(cluster_paths)),
                "representative": cluster_paths[representative_idx],
                "paths": cluster_paths,
                "center": cluster_center.tolist()
            }
            pbar.set_postfix({"size": len(cluster_paths)})
    
    # Clustering metadata
    cluster_metadata = {
        "n_clusters": int(n_clusters),
        "inertia": float(kmeans.inertia_),
        "n_iter": int(kmeans.n_iter_),
        "method": "kmeans"
    }
    
    return clusters, cluster_metadata

def get_image_paths(directory: str, recursive: bool = False) -> List[str]:
    """Get all image paths in a directory"""
    image_paths = []
    
    if recursive:
        # Count total files first for progress bar
        total_files = sum(len(files) for _, _, files in os.walk(directory))
        print(f"🔍 Scanning directory recursively ({total_files} files to check)...")
        
        with tqdm(total=total_files, desc="Scanning files", unit="file", ncols=80) as pbar:
            for root, _, files in os.walk(directory):
                for file in files:
                    pbar.update(1)
                    if os.path.splitext(file.lower())[1] in IMAGE_EXTENSIONS:
                        image_paths.append(os.path.join(root, file))
    else:
        files = os.listdir(directory)
        print(f"🔍 Scanning directory ({len(files)} files to check)...")
        
        with tqdm(files, desc="Scanning files", unit="file", ncols=80) as pbar:
            for file in pbar:
                if os.path.splitext(file.lower())[1] in IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(directory, file))
    
    print(f"📸 Found {len(image_paths)} image files")
    return image_paths

def analyze_directory(directory: str, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                     aesthetic_threshold: float = DEFAULT_AESTHETIC_THRESHOLD, recursive: bool = False,
                     skip_duplicates: bool = False, skip_aesthetics: bool = False, limit: int = 0,
                     n_clusters: Optional[int] = None) -> Dict[str, Any]:
    """Analyze a directory of images using functional approach with robust caching"""
    
    print(f"📁 Analyzing directory: {directory}")
    
    # 1. Load image paths (without limit first)
    all_image_paths = get_image_paths(directory, recursive)
    print(f"📸 Found {len(all_image_paths)} images")
    
    # 2. Generate cache key (excluding limit for stable caching)
    cache_key = generate_analysis_cache_key(directory, recursive, similarity_threshold, aesthetic_threshold, n_clusters)
    
    # 3. Try to load cached analysis results
    cached_data = load_analysis_cache(cache_key)
    if cached_data and validate_cache_integrity(cached_data, all_image_paths, n_clusters):
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
    
    def check_cache_batch(paths_batch):
        batch_cached_features = {}
        batch_cached_scores = {}
        batch_to_process = []
        
        for path in paths_batch:
            image_hash = get_image_hash(path)
            if image_hash:
                cached_data = load_cached_data(image_hash)
                
                if 'features' in cached_data and 'aesthetic_score' in cached_data:
                    batch_cached_features[path] = cached_data['features']
                    batch_cached_scores[path] = cached_data['aesthetic_score']
                else:
                    batch_to_process.append((path, image_hash))
        
        return batch_cached_features, batch_cached_scores, batch_to_process
    
    # Process cache checking in parallel
    batch_size = max(1, len(all_image_paths) // MAX_WORKERS)
    batches = [all_image_paths[i:i + batch_size] for i in range(0, len(all_image_paths), batch_size)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=len(all_image_paths), desc="Checking cache", unit="img", ncols=80) as pbar:
            futures = [executor.submit(check_cache_batch, batch) for batch in batches]
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_cached_features, batch_cached_scores, batch_to_process = future.result()
                    cached_features.update(batch_cached_features)
                    cached_scores.update(batch_cached_scores)
                    to_process.extend(batch_to_process)
                    pbar.update(len(batch_cached_features))
                except Exception as e:
                    logger.error(f"Error in cache check batch: {e}")
    
    # 6. Process new images
    features = {}
    scores = {}
    if to_process:
        print(f"🔄 Processing {len(to_process)} new images...")
        paths = [p[0] for p in to_process]
        
        # Extract features and scores in parallel
        new_features = process_images_multithreaded(paths, 'features')
        new_scores = process_images_multithreaded(paths, 'aesthetic')
        
        # Store in cache
        def save_to_cache_batch(batch_to_process):
            for (path, image_hash) in batch_to_process:
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
        
        # Save to cache in parallel
        cache_batch_size = max(1, len(to_process) // MAX_WORKERS)
        cache_batches = [to_process[i:i + cache_batch_size] for i in range(0, len(to_process), cache_batch_size)]
        
        print("💾 Saving to cache...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            with tqdm(total=len(to_process), desc="Saving cache", unit="img", ncols=80) as pbar:
                futures = [executor.submit(save_to_cache_batch, batch) for batch in cache_batches]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                        # Update progress based on batch size
                        pbar.update(cache_batch_size)
                    except Exception as e:
                        logger.error(f"Error saving cache batch: {e}")
    
    # Merge cached and new
    features.update(cached_features)
    scores.update(cached_scores)
    
    # 7. Duplicate detection
    duplicates = []
    if not skip_duplicates:
        # Pass pre-computed features to avoid redundant processing
        duplicates = find_duplicates_multithreaded(all_image_paths, threshold=similarity_threshold, pre_computed_features=features)
    
    # 8. Clustering
    clusters = {}
    cluster_features = {}
    if not skip_aesthetics:
        clusters, cluster_features = cluster_images_kmeans(all_image_paths, n_clusters)
    
    # 9. Aesthetic scoring (already done above)
    aesthetic_scores = scores
    
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
                'recursive': recursive,
                'n_clusters': n_clusters
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
    """Format analysis results for frontend display - simplified to only build images list"""
    try:
        images = []
        
        # Create lookup maps for efficient processing
        duplicate_paths = set()
        for group in results['duplicates']:
            duplicate_paths.update(group)
        
        cluster_lookup = {}
        for cluster_id, cluster_data in results['clusters'].items():
            cluster_lookup[cluster_id] = {
                'size': int(cluster_data['size']),
                'paths': set(cluster_data['paths'])
            }
        
        # Process all image paths once to build the images list
        all_paths = results.get('all_image_paths', [])
        
        if len(all_paths) > 100:  # Only show progress for large datasets
            print("📊 Formatting results for frontend...")
            with tqdm(all_paths, desc="Processing images", unit="img", ncols=80) as pbar:
                for path in pbar:
                    # Find which cluster this image belongs to
                    cluster_id = -1  # -1 indicates no cluster
                    cluster_size = 1
                    
                    for cid, cluster_info in cluster_lookup.items():
                        if path in cluster_info['paths']:
                            cluster_id = int(cid)
                            cluster_size = cluster_info['size']
                            break
                    
                    images.append({
                        'path': path,
                        'cluster': cluster_id,
                        'cluster_size': cluster_size,
                        'is_duplicate': path in duplicate_paths,
                        'is_low_aesthetic': float(results['aesthetic_scores'].get(path, 0)) < DEFAULT_AESTHETIC_THRESHOLD,
                        'aesthetic_score': float(results['aesthetic_scores'].get(path, 0))
                    })
        else:
            # Process without progress bar for small datasets
            for path in all_paths:
                # Find which cluster this image belongs to
                cluster_id = -1  # -1 indicates no cluster
                cluster_size = 1
                
                for cid, cluster_info in cluster_lookup.items():
                    if path in cluster_info['paths']:
                        cluster_id = int(cid)
                        cluster_size = cluster_info['size']
                        break
                
                images.append({
                    'path': path,
                    'cluster': cluster_id,
                    'cluster_size': cluster_size,
                    'is_duplicate': path in duplicate_paths,
                    'is_low_aesthetic': float(results['aesthetic_scores'].get(path, 0)) < DEFAULT_AESTHETIC_THRESHOLD,
                    'aesthetic_score': float(results['aesthetic_scores'].get(path, 0))
                })
        
        # Return only the images list since that's all the frontend needs
        return safe_convert_numpy({'images': images})
    except Exception as e:
        logger.error(f"Error formatting results: {str(e)}")
        return {'images': []}

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
            'limit': int(data.get('limit', 0)),
            'n_clusters': data.get('n_clusters')  # Optional, will use elbow method if not provided
        }

        # Perform analysis
        results = analyze_directory(
            directory,
            similarity_threshold=params['similarity_threshold'],
            aesthetic_threshold=params['aesthetic_threshold'],
            recursive=params['recursive'],
            skip_duplicates=params['skip_duplicates'],
            skip_aesthetics=params['skip_aesthetics'],
            limit=params['limit'],
            n_clusters=params['n_clusters']
        )

        formatted_results = format_results_for_frontend(results)
        
        # Ensure response is JSON serializable
        response = {
            'success': True,
            'images': safe_convert_numpy(formatted_results['images'])
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
        cache_key = generate_analysis_cache_key(directory, recursive, similarity_threshold, aesthetic_threshold, None)
        
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
        print(f"🚀 Wallyzer starting...")
        
        # Initialize models with progress
        initialize_models()
        
        print(f"🌐 Server running at http://localhost:{PORT}")
        print("✅ Caching system enabled")
        print("✅ Progress bars enabled for all operations")
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