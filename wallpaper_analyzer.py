# Wallpaper Analyzer V2
# Concurrency: Uses eventlet (green threads) for async I/O and Flask-SocketIO, and ThreadPoolExecutor for CPU-bound image processing. This hybrid model is robust for I/O and parallel CPU workloads.
# Configuration: All magic numbers and tunables are in the Config class below.

import os
import logging
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import multiprocessing
from functools import lru_cache
from collections import defaultdict
import sqlite3
import json
from datetime import datetime
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score
import flask
from skimage.color import rgb2lab
import open_clip
import threading
from colorlog import ColoredFormatter
import hashlib

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
    
    # Flask settings
    HOST = '0.0.0.0'
    PORT = 8000
    DEBUG = False
    
    # Database settings
    DB_PATH = 'analysis_cache.db'
    MAX_HASH_CACHE = 2000
    DUPLICATE_FEATURE_LIMIT = 1000

# Helper for pretty banners
BANNER = '\n' + '='*60 + '\n'

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(
    "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    },
))

logger = logging.getLogger("wallpaper-analyzer")
logger.setLevel(logging.INFO)
logger.handlers = []  # Remove any existing handlers
logger.addHandler(handler)
logger.propagate = False

def compute_sha256(path: str) -> str:
    """Compute SHA256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

# Database initialization
def init_db() -> None:
    conn = sqlite3.connect(Config.DB_PATH)
    c = conn.cursor()
    # Per-image cache (now hash-based)
    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash TEXT UNIQUE,
            path TEXT,
            features BLOB,
            aesthetic_score REAL,
            analyzed_at TIMESTAMP,
            other_metadata TEXT
        )
    ''')
    # Per-analysis run
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            directory TEXT,
            params TEXT,
            run_time TIMESTAMP,
            result_summary TEXT
        )
    ''')
    # Map images to analysis runs, clusters, etc.
    c.execute('''
        CREATE TABLE IF NOT EXISTS image_analysis_map (
            run_id INTEGER,
            image_id INTEGER,
            cluster_id INTEGER,
            is_duplicate BOOLEAN,
            is_low_aesthetic BOOLEAN,
            FOREIGN KEY(run_id) REFERENCES analysis_runs(id),
            FOREIGN KEY(image_id) REFERENCES images(id)
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

def get_image_cache_entry(conn, hash_: str):
    with conn:
        c = conn.cursor()
        c.execute('''
            SELECT id, features, aesthetic_score FROM images
            WHERE hash = ?
        ''', (hash_,))
        return c.fetchone()

def upsert_image_cache_entry(conn, hash_: str, path: str, features, aesthetic_score):
    c = conn.cursor()
    c.execute('''
        INSERT INTO images (hash, path, features, aesthetic_score, analyzed_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(hash) DO UPDATE SET
            path=excluded.path,
            features=excluded.features,
            aesthetic_score=excluded.aesthetic_score,
            analyzed_at=excluded.analyzed_at
    ''', (hash_, path, features, aesthetic_score, datetime.now().isoformat()))
    conn.commit()

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Global state is managed within the application context

def fast_resize_pillow(img: Image.Image, size=(224, 224)) -> Image.Image:
    """Resize image using Pillow-SIMD for best performance and quality."""
    return img.resize(size, resample=Image.Resampling.LANCZOS)

def perceptual_brightness_lab(img: Image.Image) -> float:
    """Compute perceptual brightness using LAB L channel (0-100)."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    lab = rgb2lab(arr)
    l_channel = lab[..., 0]
    return float(np.mean(l_channel))

class ImageProcessor:
    def __init__(self) -> None:
        self.device = Config.DEVICE
        self.transform = transforms.Compose([
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

    def _log_batch(self, batch_idx, total_batches, batch_paths, process_type):
        thread = threading.current_thread()
        logger.info(f"[Batch {batch_idx+1}/{total_batches}] ({len(batch_paths)} images) | {process_type.upper()} | Thread: {thread.name}")
        logger.info(f"    Files: {batch_paths[:2]}{' ...' if len(batch_paths) > 2 else ''}")

    @lru_cache(maxsize=Config.MAX_HASH_CACHE)
    def calculate_hash(self, image_path: str) -> bytes | None:
        """Calculate perceptual hash of an image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = fast_resize_pillow(image, (224, 224))
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
        total_batches = (total + batch_size - 1) // batch_size
        for i in range(0, total, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            self._log_batch(i//batch_size, total_batches, batch_paths, process_type)
            batch_images = []
            valid_paths = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image = fast_resize_pillow(image, (224, 224))
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
        
        logger.info(f"[DONE] {process_type.upper()} for {total} images in {total_batches} batches.")
        return results

    def compute_brightness(self, image_path: str) -> float:
        """Compute perceptual brightness for an image using LAB L channel."""
        try:
            image = Image.open(image_path).convert('RGB')
            image = fast_resize_pillow(image, (224, 224))
            return perceptual_brightness_lab(image)
        except Exception as e:
            logger.error(f"Error computing brightness for {image_path}: {e}")
            return -1.0

class DuplicateDetector:
    def __init__(self, image_processor: ImageProcessor) -> None:
        self.image_processor = image_processor

    def find_duplicates(self, image_paths: list[str], threshold: float = Config.DEFAULT_SIMILARITY_THRESHOLD) -> list[list[str]]:
        logger.info(BANNER + "[STEP] Duplicate Detection: Parallel Hashing and Feature Comparison" + BANNER)
        if not image_paths:
            return []

        # Phase 1: Hash-based grouping
        image_hashes = defaultdict(list)
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            logger.info(f"[ThreadPoolExecutor] MAX_WORKERS={Config.MAX_WORKERS}")
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
            logger.info(f"[Feature Phase] Running feature-based duplicate detection (limit={Config.DUPLICATE_FEATURE_LIMIT})")
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

            logger.info(f"[Feature Phase] Similarity matrix shape: {similarity_matrix.shape}")
        logger.info(f"[DONE] Duplicate detection complete. Groups found: {len(potential_duplicates)}")
        return potential_duplicates

class ImageClusterer:
    def __init__(self, image_processor: ImageProcessor) -> None:
        self.image_processor = image_processor
        self.device = Config.DEVICE
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._initialize_clip()
        
    def _initialize_clip(self) -> None:
        """Initialize CLIP model for cluster naming/tagging"""
        logger.info(f"Initializing CLIP model on device: {self.device}")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
        )
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.clip_model.eval()

    def clip_describe_image(self, image_path: str) -> str:
        """Generate a text label for an image using CLIP zero-shot classification. Returns best label, logs top-3."""
        candidate_labels = [
            # Expanded set: objects, scenes, moods, genres, colors, etc.
            "nature", "mountain", "beach", "city", "animal", "car", "anime", "abstract", "portrait", "space", "food", "flower", "building", "art", "landscape", "night", "day", "sky", "forest", "desert", "ocean", "minimal", "pattern", "macro", "technology", "sports", "fantasy", "sci-fi", "vintage", "retro",
            "people", "sunset", "sunrise", "clouds", "rain", "snow", "water", "tree", "road", "bridge", "mountains", "lake", "river", "park", "garden", "indoor", "outdoor", "architecture", "animal", "cat", "dog", "bird", "insect", "macro", "closeup", "bokeh", "colorful", "monochrome", "black and white", "pastel", "vibrant", "moody", "calm", "happy", "sad", "energetic", "relaxing", "sci-fi", "fantasy", "cyberpunk", "steampunk", "minimalist", "pattern", "texture", "illustration", "painting", "photograph", "digital art", "3d render", "graphic design"
        ]
        try:
            image = Image.open(image_path).convert('RGB')
            image = fast_resize_pillow(image, (224, 224))
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                text_inputs = self.clip_tokenizer(candidate_labels).to(self.device)
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                top_indices = similarity[0].argsort(descending=True)[:3].cpu().numpy()
                top_labels = [candidate_labels[i] for i in top_indices]
                top_scores = [float(similarity[0, i].cpu().item()) for i in top_indices]
                logger.info(f"[CLIP] Top labels for {image_path}: {list(zip(top_labels, top_scores))}")
                # Use top label if confidence > 0.15, else fallback
                if top_scores[0] > 0.15 and top_labels[0] != "unknown":
                    return top_labels[0]
                for lbl, score in zip(top_labels[1:], top_scores[1:]):
                    if score > 0.10 and lbl != "unknown":
                        return lbl
                return "Unlabeled"
        except Exception as e:
            logger.error(f"Error generating CLIP label for {image_path}: {e}")
            return "Unlabeled"
    
    def find_optimal_clusters(self, features: np.ndarray, algorithm: str = "minibatchkmeans") -> int:
        """Find optimal number of clusters using silhouette analysis"""
        best_score = -1
        best_n_clusters = Config.MIN_CLUSTERS
        for n_clusters in range(Config.MIN_CLUSTERS, min(Config.MAX_CLUSTERS + 1, len(features))):
            if algorithm == "minibatchkmeans":
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(features, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
        return best_n_clusters
    
    def cluster_images(self, image_paths: list[str], algorithm: str = "minibatchkmeans", n_clusters: str | int = "auto", features: dict = None) -> tuple[dict, dict]:
        logger.info(BANNER + f"[STEP] Clustering: {algorithm.upper()}" + BANNER)
        if features is None:
            features_arr, valid_paths = self.extract_features(image_paths)
        else:
            # features is a dict: {path: feature}
            valid_paths = [p for p in image_paths if p in features]
            features_arr = np.stack([features[p] for p in valid_paths]) if valid_paths else np.array([])
        if len(features_arr) < Config.MIN_CLUSTERS:
            return {0: {"paths": valid_paths, "size": len(valid_paths), "label": "unknown"}}, {}
        clusters = {}
        if algorithm == "dbscan":
            logger.info(f"[DBSCAN] eps=0.5, min_samples=3, features shape={features_arr.shape}")
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            cluster_labels = dbscan.fit_predict(features_arr)
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label == -1:
                    continue  # noise
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_paths = [valid_paths[idx] for idx in cluster_indices]
                cluster_features = features_arr[cluster_indices]
                # Use first image as representative
                representative_idx = 0
                representative_path = cluster_paths[representative_idx]
                cluster_label = self.clip_describe_image(representative_path)
                clusters[label] = {
                    "size": len(cluster_paths),
                    "representative": representative_path,
                    "paths": cluster_paths,
                    "label": cluster_label
                }
            logger.info(f"[DBSCAN] Clusters found: {len(clusters)}")
            return clusters, {}
        else:
            # MiniBatchKMeans (default)
            if n_clusters == "auto":
                n_clusters_val = self.find_optimal_clusters(features_arr, algorithm="minibatchkmeans")
            else:
                n_clusters_val = int(n_clusters)
            logger.info(f"[MiniBatchKMeans] n_clusters={n_clusters_val}, features shape={features_arr.shape}")
            kmeans = MiniBatchKMeans(n_clusters=n_clusters_val, random_state=42)
            cluster_labels = kmeans.fit_predict(features_arr)
            for i in range(n_clusters_val):
                cluster_indices = np.where(cluster_labels == i)[0]
                cluster_paths = [valid_paths[idx] for idx in cluster_indices]
                cluster_features = features_arr[cluster_indices]
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_features - center, axis=1)
                representative_idx = np.argmin(distances)
                representative_path = cluster_paths[representative_idx]
                cluster_label = self.clip_describe_image(representative_path)
                clusters[i] = {
                    "size": len(cluster_paths),
                    "representative": representative_path,
                    "paths": cluster_paths,
                    "label": cluster_label
                }
            logger.info(f"[MiniBatchKMeans] Clusters found: {len(clusters)}")
        return clusters, {}

class WallpaperAnalyzer:
    def __init__(self) -> None:
        self.image_processor = ImageProcessor()
        self.duplicate_detector = DuplicateDetector(self.image_processor)
        self.image_clusterer = ImageClusterer(self.image_processor)

    def analyze_directory(self, directory: str, similarity_threshold: float = Config.DEFAULT_SIMILARITY_THRESHOLD,
                         aesthetic_threshold: float = Config.DEFAULT_AESTHETIC_THRESHOLD, recursive: bool = False,
                         skip_duplicates: bool = False, skip_aesthetics: bool = False, limit: int = 0,
                         cluster_algorithm: str = "minibatchkmeans", n_clusters: str | int = "auto") -> dict:
        import time
        t0 = time.time()
        logger.info(BANNER + f"[ANALYSIS START] Directory: {directory}" + BANNER)
        logger.info(f"[CONFIG] Device: {Config.DEVICE}, Batch Size: {Config.BATCH_SIZE}, Max Workers: {Config.MAX_WORKERS}")
        logger.info(f"[PARAMS] similarity_threshold={similarity_threshold}, aesthetic_threshold={aesthetic_threshold}, recursive={recursive}, skip_duplicates={skip_duplicates}, skip_aesthetics={skip_aesthetics}, limit={limit}, cluster_algorithm={cluster_algorithm}, n_clusters={n_clusters}")
        total_steps = 5
        current_step = 0
        logger.info(f"Starting analysis for {directory}")
        # 1. Load image paths
        logger.info(BANNER + "[STEP] Image Loading & Caching" + BANNER)
        logger.info("Loading image paths...")
        image_paths = self._get_image_paths(directory, recursive)
        if limit and limit > 0:
            image_paths = image_paths[:limit]
        logger.info(f"Loaded {len(image_paths)} image paths.")
        # Per-image cache check (now hash-based)
        conn = sqlite3.connect(Config.DB_PATH)
        cached_features = {}
        cached_scores = {}
        to_process = []
        image_hashes = {}
        for path in image_paths:
            try:
                hash_ = compute_sha256(path)
                image_hashes[path] = hash_
                entry = get_image_cache_entry(conn, hash_)
                if entry and entry[1] is not None:
                    cached_features[path] = np.frombuffer(entry[1], dtype=np.float32)
                    if entry[2] is not None:
                        cached_scores[path] = entry[2]
                else:
                    to_process.append((path, hash_))
            except Exception as e:
                logger.error(f"Error checking cache for {path}: {e}")
                to_process.append((path, None))
        # Only process new/changed images
        features = {}
        scores = {}
        brightness_scores = {}
        if to_process:
            logger.info(f"[CACHE] {len(to_process)} images to process (not cached)")
            paths = [p[0] for p in to_process if p[1] is not None]
            # Feature extraction
            new_features = self.image_processor.process_batch(paths, process_type='features')
            # Aesthetic scoring
            new_scores = self.image_processor.process_batch(paths, process_type='aesthetic')
            # Brightness scoring
            for path in paths:
                brightness_scores[path] = self.image_processor.compute_brightness(path)
            # Store in DB
            for i, (path, hash_) in enumerate(to_process):
                if not hash_:
                    continue
                feat = new_features.get(path)
                score = new_scores.get(path)
                if feat is not None:
                    features[path] = feat
                if score is not None:
                    scores[path] = score
                upsert_image_cache_entry(conn, hash_, path,
                    np.array(feat, dtype=np.float32).tobytes() if feat is not None else None,
                    float(score) if score is not None else None)
        # Merge cached and new
        features.update(cached_features)
        scores.update(cached_scores)
        # conn.close()  # <-- Do NOT close here! Keep open for all DB ops
        current_step += 1
        t1 = time.time()
        logger.info(BANNER + "[STEP] Feature Extraction" + BANNER)
        # 2. Feature extraction (removed redundant recomputation)
        logger.info("Feature extraction complete.")
        current_step += 1
        t2 = time.time()
        logger.info(BANNER + "[STEP] Duplicate Detection" + BANNER)
        # 3. Duplicate detection
        logger.info("Starting duplicate detection...")
        duplicates = []
        if not skip_duplicates:
            # Use features dict for duplicate detection if possible
            duplicates = self.duplicate_detector.find_duplicates(image_paths, threshold=similarity_threshold)
        logger.info("Duplicate detection complete.")
        current_step += 1
        t3 = time.time()
        logger.info(BANNER + "[STEP] Clustering" + BANNER)
        # 4. Clustering
        logger.info("Starting clustering...")
        clusters = {}
        cluster_features = {}
        if not skip_aesthetics:
            # Use features dict for clustering if possible
            cluster_result = self.image_clusterer.cluster_images(image_paths, algorithm=cluster_algorithm, n_clusters=n_clusters, features=features)
            if isinstance(cluster_result, tuple):
                clusters, cluster_features = cluster_result
            else:
                clusters = cluster_result
        logger.info("Clustering complete.")
        current_step += 1
        t4 = time.time()
        logger.info(BANNER + "[STEP] Aesthetic Scoring & Brightness" + BANNER)
        # 5. Aesthetic scoring (removed redundant recomputation)
        logger.info("Aesthetic scoring complete.")
        # Compute brightness for all images (if not already done)
        for path in image_paths:
            if path not in brightness_scores:
                brightness_scores[path] = self.image_processor.compute_brightness(path)
        current_step += 1
        t5 = time.time()
        logger.info(BANNER + f"[ANALYSIS COMPLETE] Total time: {t5-t0:.2f}s" + BANNER)
        logger.info(f"[TIMING] image loading: {t1-t0:.2f}s, feature extraction: {t2-t1:.2f}s, duplicate detection: {t3-t2:.2f}s, clustering: {t4-t3:.2f}s, scoring: {t5-t4:.2f}s")
        # After all analysis steps (duplicates, clusters, scores) are complete:
        # Insert analysis run
        # Clamp scores to [0, 1] for frontend
        aesthetic_scores = {path: max(0.0, min(1.0, float(score))) for path, score in scores.items()}
        result_summary = {
            'duplicates': duplicates,
            'clusters': clusters,
            'aesthetic_scores': aesthetic_scores,
            'brightness_scores': brightness_scores
        }
        run_id = insert_analysis_run(conn, directory, {
            'similarity_threshold': similarity_threshold,
            'aesthetic_threshold': aesthetic_threshold,
            'recursive': recursive,
            'skip_duplicates': skip_duplicates,
            'skip_aesthetics': skip_aesthetics,
            'limit': limit,
            'cluster_algorithm': cluster_algorithm,
            'n_clusters': n_clusters
        }, result_summary)
        # Map images to this run
        for path in image_paths:
            try:
                hash_ = image_hashes.get(path)
                if not hash_:
                    continue
                image_id = get_image_id_by_hash(conn, hash_)
                if image_id is None:
                    continue
                # Find cluster_id
                cluster_id = -1
                for cid, cdata in clusters.items():
                    if 'paths' in cdata and path in cdata['paths']:
                        cluster_id = cid
                        break
                # Is duplicate?
                is_dup = any(path in group for group in duplicates)
                # Is low aesthetic?
                is_low = False
                if path in aesthetic_scores:
                    is_low = aesthetic_scores[path] < aesthetic_threshold
                insert_image_analysis_map(conn, run_id, image_id, cluster_id, is_dup, is_low)
            except Exception as e:
                logger.error(f"Error mapping image to analysis run: {e}")
        conn.close()  # <-- Now close after all DB ops
        # Format results directly for the frontend
        frontend_images = []
        for path in image_paths:
            cluster_id = -1
            for cid, cdata in clusters.items():
                if 'paths' in cdata and path in cdata['paths']:
                    cluster_id = cid
                    break
            
            frontend_images.append({
                'path': path,
                'cluster': cluster_id,
                'is_duplicate': any(path in group for group in duplicates),
                'is_low_aesthetic': scores.get(path, 0) < aesthetic_threshold,
                'aesthetic_score': scores.get(path),
                'brightness': brightness_scores.get(path)
            })
        
        return {
            'success': True,
            'images': frontend_images,
            'clusters': clusters,
            'duplicates': duplicates
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
            'limit': int(data.get('limit', 0)),
            'cluster_algorithm': data.get('cluster_algorithm', 'minibatchkmeans'),
            'n_clusters': data.get('n_clusters', 'auto')
        }
        logger.info(f"Analysis parameters: {params}")
        analyzer = WallpaperAnalyzer()
        results = analyzer.analyze_directory(
            directory,
            similarity_threshold=params['similarity_threshold'],
            aesthetic_threshold=params['aesthetic_threshold'],
            recursive=params['recursive'],
            skip_duplicates=params['skip_duplicates'],
            skip_aesthetics=params['skip_aesthetics'],
            limit=params['limit'],
            cluster_algorithm=params['cluster_algorithm'],
            n_clusters=params['n_clusters']
        )
        logger.info("Analysis completed")
        return jsonify(results)
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

def insert_analysis_run(conn, directory: str, params: dict, result_summary: dict) -> int:
    c = conn.cursor()
    c.execute('''
        INSERT INTO analysis_runs (directory, params, run_time, result_summary)
        VALUES (?, ?, ?, ?)
    ''', (directory, json.dumps(params), datetime.now().isoformat(), json.dumps(result_summary)))
    conn.commit()
    return c.lastrowid

def insert_image_analysis_map(conn, run_id: int, image_id: int, cluster_id: int, is_duplicate: bool, is_low_aesthetic: bool):
    c = conn.cursor()
    c.execute('''
        INSERT INTO image_analysis_map (run_id, image_id, cluster_id, is_duplicate, is_low_aesthetic)
        VALUES (?, ?, ?, ?, ?)
    ''', (run_id, image_id, cluster_id, int(is_duplicate), int(is_low_aesthetic)))
    conn.commit()

def get_image_id_by_hash(conn, hash_: str) -> int | None:
    c = conn.cursor()
    c.execute('SELECT id FROM images WHERE hash = ?', (hash_,))
    row = c.fetchone()
    return row[0] if row else None

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG) 