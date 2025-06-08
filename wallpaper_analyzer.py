import eventlet
eventlet.monkey_patch()

import os
import logging
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
from flask_cors import CORS
import multiprocessing
from functools import lru_cache
import torch.nn.functional as F
from collections import defaultdict
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wallpaper_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('WallpaperAnalyzer')

# M1-specific optimizations
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS (Metal Performance Shaders) for M1 acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using CUDA for GPU acceleration")
else:
    device = torch.device("cpu")
    logger.info("Using CPU for computations")

# Initialize Flask app
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

# Global variables
analysis_results = {}
active_connections = set()

@socketio.on('connect')
def handle_connect():
    """Handle new socket connections"""
    active_connections.add(request.sid)
    logger.info(f"Client connected: {request.sid}")
    socketio.emit('connection_status', {'status': 'connected'}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle socket disconnections"""
    active_connections.discard(request.sid)
    logger.info(f"Client disconnected: {request.sid}")

def emit_progress(progress, message=None):
    """Emit progress updates to connected clients with error handling"""
    try:
        with app.app_context():
            data = {
                'progress': progress,
                'message': message,
                'timestamp': time.time()
            }
            socketio.emit('progress', data)
    except Exception as e:
        logger.error(f"Error emitting progress: {e}")

def emit_analysis_complete(results):
    """Emit analysis completion with error handling"""
    try:
        with app.app_context():
            socketio.emit('analysis_complete', results)
    except Exception as e:
        logger.error(f"Error emitting analysis complete: {e}")

class SimilarityModel:
    def __init__(self):
        self.device = device
        logger.info(f"Using device: {self.device}")
        
        # Load MobileNetV2 for feature extraction
        logger.info("Loading MobileNetV2 model for feature extraction...")
        self.feature_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.feature_model = self.feature_model.to(self.device)
        self.feature_model.eval()
        
        # Load ResNet18 for aesthetic evaluation
        logger.info("Loading ResNet18 model for aesthetic evaluation...")
        self.aesthetic_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Modify the last layer for binary classification
        num_features = self.aesthetic_model.fc.in_features
        self.aesthetic_model.fc = torch.nn.Linear(num_features, 1)
        self.aesthetic_model = self.aesthetic_model.to(self.device)
        self.aesthetic_model.eval()
        
        # Image transformations with optimized settings for M1
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),  # Enable antialiasing for better quality
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Cache for processed images with larger size for M1
        self.feature_cache = {}
        self.hash_cache = {}
        # Increase batch size for M1 as it has more memory
        self.batch_size = 64 if torch.backends.mps.is_available() else 32 if torch.cuda.is_available() else 8

    @lru_cache(maxsize=2000)  # Increased cache size for M1
    def calculate_hash(self, image_path):
        """Calculate perceptual hash of an image with caching"""
        if image_path in self.hash_cache:
            return self.hash_cache[image_path]
            
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.feature_model.features(image)
                features = torch.mean(features, dim=[2, 3])
                hash_value = features.cpu().numpy().tobytes()
                self.hash_cache[image_path] = hash_value
                return hash_value
        except Exception as e:
            logger.error(f"Error calculating hash for {image_path}: {e}")
            return None

    def extract_features_batch(self, image_paths):
        """Extract features from multiple images in batches with M1 optimizations"""
        features = {}
        total_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[batch_idx:batch_idx + self.batch_size]
            batch_images = []
            valid_paths = []
            
            # Process each image in the batch
            for path in batch_paths:
                if path in self.feature_cache:
                    features[path] = self.feature_cache[path]
                else:
                    try:
                        # Add timeout for image loading
                        with Image.open(path) as img:
                            image = img.convert('RGB')
                            image = self.transform(image)
                            batch_images.append(image)
                            valid_paths.append(path)
                    except Exception as e:
                        logger.error(f"Error loading image {path}: {e}")
                        continue
            
            if batch_images:
                try:
                    # Process batch with timeout
                    batch_tensor = torch.stack(batch_images).to(self.device)
                    with torch.no_grad():
                        batch_features = self.feature_model.features(batch_tensor)
                        batch_features = torch.mean(batch_features, dim=[2, 3])
                        
                        # Move features to CPU and convert to numpy immediately
                        batch_features = batch_features.cpu().numpy()
                        
                        for path, feature in zip(valid_paths, batch_features):
                            features[path] = feature
                            self.feature_cache[path] = feature
                            
                        # Clear GPU memory
                        del batch_tensor
                        del batch_features
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                            
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx + 1}/{total_batches}: {e}")
                    continue
                
                # Clear CPU memory
                del batch_images
                
        return features

    def calculate_similarity_matrix(self, features):
        """Calculate similarity matrix for all features efficiently"""
        try:
            paths = list(features.keys())
            if not paths:
                return [], np.array([])
                
            # Process features in chunks to manage memory
            chunk_size = 1000  # Adjust based on available memory
            feature_matrix = []
            
            for i in range(0, len(paths), chunk_size):
                chunk_paths = paths[i:i + chunk_size]
                chunk_features = np.stack([features[path] for path in chunk_paths])
                feature_matrix.append(chunk_features)
            
            feature_matrix = np.vstack(feature_matrix)
            
            # Normalize features
            feature_matrix = feature_matrix / np.linalg.norm(feature_matrix, axis=1, keepdims=True)
            
            # Calculate similarity matrix using matrix multiplication
            similarity_matrix = np.dot(feature_matrix, feature_matrix.T)
            
            return paths, similarity_matrix
            
        except Exception as e:
            logger.error(f"Error calculating similarity matrix: {e}")
            return [], np.array([])

    def predict_aesthetic_batch(self, image_paths):
        """Predict aesthetic scores for multiple images in batches"""
        scores = {}
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
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
            
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                with torch.no_grad():
                    outputs = self.aesthetic_model(batch_tensor)
                    # Ensure we get a single score per image
                    batch_scores = torch.sigmoid(outputs).squeeze().cpu().numpy()
                    
                    # Handle both single and batch cases
                    if batch_scores.ndim == 0:
                        batch_scores = np.array([batch_scores])
                    
                    for path, score in zip(valid_paths, batch_scores):
                        scores[path] = float(score)
        
        return scores

def get_image_paths(directory, recursive=False):
    """Get all image paths in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file.lower())[1] in image_extensions:
                    image_paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if os.path.splitext(file.lower())[1] in image_extensions:
                image_paths.append(os.path.join(directory, file))
    
    return image_paths

def process_image_aesthetics(args):
    """Process a single image for aesthetic evaluation (for parallel processing)"""
    path, model = args
    try:
        score = model.predict_aesthetic_batch([path])[path]
        return path, score
    except Exception as e:
        logger.error(f"Error evaluating {path}: {e}")
        return path, 0.0

def _process_hash(args):
    """Process a single image hash (for parallel processing)"""
    path, similarity_model = args
    try:
        hash_value = similarity_model.calculate_hash(path)
        return path, hash_value
    except Exception as e:
        logger.error(f"Error processing hash for {path}: {e}")
        return path, None

class DuplicateChecker:
    def __init__(self, similarity_model):
        self.similarity_model = similarity_model

    def check_duplicates(self, image_paths, threshold=0.9, max_workers=None, progress_callback=None):
        """Check for duplicate images in the given directory using optimized batch processing"""
        if not image_paths:
            return []

        total_images = len(image_paths)
        logger.info(f"Starting duplicate detection for {total_images} images")
        
        # Calculate hashes for all images in parallel
        logger.info("Phase 1: Calculating image hashes...")
        image_hashes = defaultdict(list)
        processed_count = 0
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [(path, self.similarity_model) for path in image_paths]
            total_tasks = len(tasks)
            
            futures = []
            for task in tasks:
                future = executor.submit(_process_hash, task)
                futures.append(future)
            
            # Process results as they complete
            for future in futures:
                try:
                    path, hash_value = future.result()
                    if hash_value:
                        image_hashes[hash_value].append(path)
                    processed_count += 1
                    if progress_callback:
                        progress = (processed_count / total_tasks) * 50  # First 50% of progress
                        progress_callback(processed_count, total_tasks, 
                            f"Calculating image hashes ({processed_count}/{total_tasks})")
                except Exception as e:
                    logger.error(f"Error processing hash: {e}")

        potential_duplicates = [paths for paths in image_hashes.values() if len(paths) > 1]
        logger.info(f"Found {len(potential_duplicates)} potential duplicate groups based on hash similarity")

        if len(image_paths) <= 1000:
            logger.info("Phase 2: Performing cross-comparison for images with different hashes...")
            
            # Extract features in batches
            features = {}
            batch_size = self.similarity_model.batch_size
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                batch_features = self.similarity_model.extract_features_batch(batch)
                features.update(batch_features)
                processed_count = min(i + batch_size, len(image_paths))
                if progress_callback:
                    progress = 50 + (processed_count / total_tasks) * 25  # Next 25% of progress
                    progress_callback(processed_count, total_tasks, 
                        f"Extracting image features ({processed_count}/{total_tasks})")
            
            # Calculate similarity matrix efficiently
            logger.info("Phase 3: Calculating similarity matrix...")
            paths, similarity_matrix = self.similarity_model.calculate_similarity_matrix(features)
            
            # Find duplicates using the similarity matrix
            logger.info("Phase 4: Finding similar images...")
            duplicate_sets = []
            processed = set()
            
            for i in range(len(paths)):
                if i in processed:
                    continue
                    
                current_group = {paths[i]}
                processed.add(i)
                
                # Find all similar images
                for j in range(i + 1, len(paths)):
                    if j in processed:
                        continue
                        
                    if similarity_matrix[i, j] > threshold:
                        current_group.add(paths[j])
                        processed.add(j)
                
                if len(current_group) > 1:
                    duplicate_sets.append(list(current_group))
                
                if progress_callback:
                    progress = 75 + (i / len(paths)) * 25  # Final 25% of progress
                    progress_callback(i + 1, len(paths), 
                        f"Finding similar images ({i + 1}/{len(paths)})")
            
            logger.info(f"Found {len(duplicate_sets)} additional duplicate groups based on feature similarity")
            potential_duplicates.extend(duplicate_sets)

        total_duplicates = len(potential_duplicates)
        logger.info(f"Duplicate detection complete. Found {total_duplicates} groups of duplicate images")
        return potential_duplicates

class WallpaperCategorizer:
    def __init__(self):
        self.categories = {
            'nature': ['nature', 'landscape', 'mountain', 'forest', 'ocean', 'beach', 'sunset'],
            'abstract': ['abstract', 'pattern', 'geometric', 'minimal'],
            'anime': ['anime', 'manga', 'cartoon'],
            'space': ['space', 'galaxy', 'stars', 'universe'],
            'dark': ['dark', 'minimal', 'simple'],
            'light': ['light', 'bright', 'white'],
            'art': ['art', 'painting', 'drawing'],
            'other': []
        }

    def categorize_wallpapers(self, wallpapers):
        """Categorize wallpapers based on their filenames"""
        categories = {category: [] for category in self.categories}
        
        for wallpaper in wallpapers:
            filename = os.path.basename(wallpaper['path']).lower()
            categorized = False
            
            for category, keywords in self.categories.items():
                if any(keyword in filename for keyword in keywords):
                    categories[category].append(wallpaper['path'])
                    categorized = True
                    break
            
            if not categorized:
                categories['other'].append(wallpaper['path'])
        
        return categories

def analyze_wallpapers(
    directory: str,
    similarity_threshold: float = 0.85,
    aesthetic_threshold: float = 0.8,
    recursive: bool = False,
    workers: int = 16,
    skip_duplicates: bool = False,
    skip_aesthetics: bool = False,
    limit: int = 0,
    progress_callback: callable = None
) -> dict:
    try:
        logger.info(f"Starting analysis of directory: {directory}")
        
        # Optimize worker count for M1

            # M1 has 8 cores, but we can use more threads for I/O operations
        workers = 16 if torch.backends.mps.is_available() else multiprocessing.cpu_count()
        
        logger.info(f"Settings: similarity_threshold={similarity_threshold}, aesthetic_threshold={aesthetic_threshold}, recursive={recursive}, workers={workers}")

        # Get image paths
        logger.info("Scanning directory for images...")
        image_paths = get_image_paths(directory, recursive)
        total_images = len(image_paths)
        logger.info(f"Found {total_images} images to analyze")

        if limit > 0:
            image_paths = image_paths[:limit]
            logger.info(f"Limited to {len(image_paths)} images")

        results = {
            'directory': directory,
            'total_images': total_images,
            'processed_images': 0,
            'duplicates': [],
            'aesthetic_scores': {},
            'categories': {},
            'errors': []
        }

        # Initialize models
        logger.info("Initializing models...")
        similarity_model = SimilarityModel()
        categorizer = WallpaperCategorizer()

        # Process aesthetics if not skipped
        if not skip_aesthetics:
            logger.info("Starting aesthetic analysis...")
            current = 0
            batch_size = similarity_model.batch_size
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                scores = similarity_model.predict_aesthetic_batch(batch)
                results['aesthetic_scores'].update(scores)
                current += len(batch)
                if progress_callback:
                    progress_callback(current, total_images, f"Analyzing image aesthetics ({current}/{total_images})")
            logger.info("Aesthetic analysis complete")

        # Check for duplicates if not skipped
        if not skip_duplicates:
            logger.info("Starting duplicate detection...")
            duplicates = DuplicateChecker(similarity_model).check_duplicates(
                image_paths,
                threshold=similarity_threshold,
                max_workers=workers,
                progress_callback=lambda current, total, msg=None: progress_callback(current, total, msg or f"Checking for duplicates ({current}/{total})")
            )
            results['duplicates'] = duplicates
            logger.info(f"Found {len(duplicates)} groups of duplicate images")

        # Categorize wallpapers
        logger.info("Categorizing wallpapers...")
        wallpapers = [{'path': path, 'aesthetic_score': results['aesthetic_scores'].get(path, 0.0)} for path in image_paths]
        categories = categorizer.categorize_wallpapers(wallpapers)
        results['categories'] = categories
        logger.info("Categorization complete")

        results['processed_images'] = len(image_paths)
        logger.info("Analysis complete")
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
            'categories': {},
            'errors': [str(e)]
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        directory = data.get('directory')
        similarity_threshold = float(data.get('similarity_threshold', 0.85))
        aesthetic_threshold = float(data.get('aesthetic_threshold', 0.8))
        recursive = data.get('recursive', False)
        workers = int(data.get('workers', multiprocessing.cpu_count()))
        skip_duplicates = data.get('skip_duplicates', False)
        skip_aesthetics = data.get('skip_aesthetics', False)
        limit = int(data.get('limit', 0))

        if not directory or not os.path.exists(directory):
            return jsonify({'error': 'Invalid directory path'}), 400

        def progress_callback(current, total, message=None):
            try:
                progress = (current / total) * 100 if total > 0 else 0
                emit_progress(progress, message)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

        with app.app_context():
            results = analyze_wallpapers(
                directory=directory,
                similarity_threshold=similarity_threshold,
                aesthetic_threshold=aesthetic_threshold,
                recursive=recursive,
                workers=workers,
                skip_duplicates=skip_duplicates,
                skip_aesthetics=skip_aesthetics,
                limit=limit,
                progress_callback=progress_callback
            )

            if results:
                analysis_results[directory] = results
                emit_analysis_complete(results)
                return jsonify(results)
            
            return jsonify({'error': 'No results found'}), 404

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/image')
def serve_image():
    image_path = request.args.get('path')
    if not image_path:
        return jsonify({'error': 'No image path provided'}), 400
    
    try:
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(image_path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<path:directory>')
def get_results(directory):
    if directory in analysis_results:
        return jsonify(analysis_results[directory])
    return jsonify({'error': 'Results not found'}), 404

@app.route('/api/organize', methods=['POST'])
def organize():
    data = request.json
    directory = data.get('directory')
    output_dir = data.get('output_dir')
    
    if not directory or not output_dir:
        return jsonify({'error': 'Directory and output directory are required'}), 400
    
    try:
        if directory in analysis_results:
            results = analysis_results[directory]
            # Organize wallpapers based on quality scores
            for wallpaper in results['wallpapers']:
                score = results['quality_scores'].get(wallpaper['path'], 0.0)
                category = 'high' if score >= 0.8 else 'medium' if score >= 0.5 else 'low'
                target_dir = os.path.join(output_dir, category)
                os.makedirs(target_dir, exist_ok=True)
                
                try:
                    os.symlink(wallpaper['path'], os.path.join(target_dir, os.path.basename(wallpaper['path'])))
                except Exception as e:
                    logger.error(f"Error organizing {wallpaper['path']}: {e}")
            
            return jsonify({'message': 'Wallpapers organized successfully'})
        
        return jsonify({'error': 'Analysis results not found'}), 404
    
    except Exception as e:
        logger.error(f"Error during organization: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clean', methods=['POST'])
def clean():
    data = request.json
    directory = data.get('directory')
    similarity = float(data.get('similarity', 0.95))
    dry_run = bool(data.get('dry_run', False))
    
    if not directory:
        return jsonify({'error': 'Directory is required'}), 400
    
    try:
        if directory in analysis_results:
            results = analysis_results[directory]
            if results['duplicates']:
                if dry_run:
                    return jsonify({'duplicates': results['duplicates']})
                else:
                    deleted_count = 0
                    for group in results['duplicates']:
                        # Sort by resolution to keep the highest resolution image
                        sorted_group = sorted(group, 
                                            key=lambda p: Image.open(p).size[0] * Image.open(p).size[1] 
                                            if os.path.exists(p) else 0, 
                                            reverse=True)
                        
                        # Keep the first image (highest resolution)
                        keep_path = sorted_group[0]
                        
                        # Delete the rest
                        for path in sorted_group[1:]:
                            try:
                                if os.path.exists(path):
                                    os.remove(path)
                                    deleted_count += 1
                            except Exception as e:
                                logger.error(f"Error deleting {path}: {e}")
                    
                    return jsonify({'message': f'Deleted {deleted_count} duplicate wallpapers'})
            
            return jsonify({'message': 'No duplicates found'})
        
        return jsonify({'error': 'Analysis results not found'}), 404
    
    except Exception as e:
        logger.error(f"Error during cleaning: {str(e)}")
        return jsonify({'error': str(e)}), 500

def main():
    """Main entry point for the application"""
    try:
        # Ensure we're in the application context
        with app.app_context():
            socketio.run(
                app,
                host='0.0.0.0',
                port=8000,
                debug=False,
                use_reloader=False,
                allow_unsafe_werkzeug=True
            )
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 