# 🎨 Wallpaper Analyzer V2

A powerful and intelligent wallpaper management system that helps you organize, analyze, and optimize your wallpaper collection using advanced AI and computer vision techniques.

![Wallpaper Analyzer](https://img.shields.io/badge/Wallpaper-Analyzer-blue)
![Python](https://img.shields.io/badge/Python-3%2E8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- 🔍 **Smart Duplicate Detection**: Find and manage duplicate wallpapers using perceptual hashing and deep learning
- 🎯 **Aesthetic Scoring**: AI-powered evaluation of wallpaper aesthetics
- 🚀 **High Performance**: Optimized for CPU, CUDA, and Apple Silicon (MPS) processing
- 🌐 **Web Interface**: Beautiful and responsive web UI for easy interaction
- 📊 **Real-time Progress**: Live progress updates during analysis via WebSocket
- 🔄 **Batch Processing**: Efficient batch processing of large image collections
- 🎨 **Image Analysis**: Advanced feature extraction using ResNet50
- 📱 **Cross-Platform**: Works on Windows, macOS, and Linux
- 💾 **Result Caching**: Intelligent caching system for faster repeated analyses
- 📈 **Image Clustering**: Automatic grouping of similar wallpapers using user-selectable MiniBatchKMeans or DBSCAN
- 🏷️ **CLIP-based Cluster Naming**: Each cluster is automatically tagged with a descriptive label using OpenAI CLIP
- 💡 **Perceptual Brightness**: Each image is scored for perceptual brightness using LAB color space
- ⚡ **Performance Optimization**: Uses Pillow-SIMD for fast image loading/resizing, automatic device selection (CPU/CUDA/MPS), and batch size adjustment
- 🔄 **Recursive Analysis**: Option to analyze subdirectories
- 🧑‍💻 **User-Configurable**: Choose clustering algorithm, cluster count, thresholds, and more in the UI

## 🛠️ Technical Stack

- **Backend**: Python 3.8+
- **Web Framework**: Flask with SocketIO
- **AI/ML**: PyTorch, torchvision, open_clip_torch (CLIP)
- **Image Processing**: Pillow-SIMD, OpenCV, scikit-image
- **Data Processing**: NumPy, SciPy, scikit-learn
- **Real-time Communication**: Flask-SocketIO, eventlet
- **Image Analysis**: ResNet50, Perceptual Hashing, CLIP
- **Database**: SQLite for result caching
- **Frontend**: Next.js, React, Tailwind CSS, shadcn/ui

## 📋 Requirements

- Python 3.8 or higher
- Node.js 18+ (for frontend)
- CUDA-capable GPU (optional, for faster processing)
- MPS support for Apple Silicon (optional, recommended for Mac)
- Pillow-SIMD (install via Homebrew or pip for best performance)
- open_clip_torch (for CLIP-based cluster naming)

## 🚀 Installation

### Backend

1. Clone the repository:

```bash
git clone https://github.com/yourusername/wallpaper-analyzerV2.git
cd wallpaper-analyzerV2
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional, for best performance on Mac):

```bash
brew install pillow-simd
pip uninstall pillow && pip install --force-reinstall pillow-simd
```

5. (Optional, for Apple Silicon):

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
# or follow PyTorch MPS install instructions
```

### Frontend

1. Go to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
pnpm install  # or npm install or yarn install
```

3. Start the development server:

```bash
pnpm dev  # or npm run dev or yarn dev
```

## 💻 Usage

### Start the Backend

```bash
python wallpaper_analyzer.py
```

The backend will start on [http://localhost:8000](http://localhost:8000)

### Start the Frontend

```bash
cd frontend
pnpm dev  # or npm run dev
```

The frontend will start on [http://localhost:3000](http://localhost:3000)

### Analyze Wallpapers

1. Open your browser and go to [http://localhost:3000](http://localhost:3000)
2. Select a directory containing your wallpapers and start the analysis!
3. Choose clustering algorithm (MiniBatchKMeans or DBSCAN), cluster count, and thresholds in the settings panel.
4. View clusters, CLIP tags, brightness, and more in the UI.

## ⚙️ Configuration

The application can be configured through the `Config` class in `wallpaper_analyzer.py`:

- `DEVICE`: Processing device (CPU/CUDA/MPS)
- `BATCH_SIZE`: Number of images processed simultaneously (auto-adjusts based on device)
- `MAX_WORKERS`: Maximum number of parallel workers (auto-adjusts based on device)
- `DEFAULT_SIMILARITY_THRESHOLD`: Threshold for duplicate detection (default: 0.85)
- `DEFAULT_AESTHETIC_THRESHOLD`: Threshold for aesthetic scoring (default: 0.8)
- `MIN_CLUSTERS`/`MAX_CLUSTERS`: Range for image clustering (default: 5-10)
- `CLUSTER_FEATURE_DIM`: Feature dimension for clustering (default: 2048)
- `HOST`/`PORT`: Server host/port (default: 0.0.0.0:8000)
- `DEBUG`: Debug mode flag
- `ENABLE_APP_LOGGING`: Application logging flag
- `ENABLE_SOCKET_LOGGING`: Socket.IO logging flag
- `DB_PATH`: Path to SQLite database for caching

You can also use environment variables for deployment flexibility (recommended for production).

## 🧵 Concurrency Model

- The backend uses **eventlet** (green threads) for async I/O and concurrency, which is best for I/O-bound workloads and works well with Flask-SocketIO.
- For CPU-bound tasks (like image processing), a **ThreadPoolExecutor** is used for parallelism. This hybrid approach is robust for most workloads.
- If you want pure CPU-bound scaling, consider using multiprocessing, but eventlet is recommended for this web+I/O use case.

## 🗂️ Project Structure

```
wallpaper-analyzerV2/
├── wallpaper_analyzer.py      # Backend server and analysis logic
├── requirements.txt          # Python dependencies
├── frontend/                 # Next.js/React frontend
│   ├── app/                  # Main app pages
│   ├── components/           # React components (shadcn/ui, custom)
│   ├── hooks/                # Custom React hooks
│   ├── lib/                  # Utilities
│   └── ...
├── analysis_cache.db         # SQLite cache
├── analyzed.log              # Log file
├── README.md                 # This file
├── index.html                # Legacy HTML UI (for backup/testing only)
└── ...
```

## 🧪 Testing & Security

- Run `pip-audit` to check for Python dependency vulnerabilities.
- Use `pnpm audit` or `npm audit` in the frontend for JS dependency security.
- Enable ESLint and TypeScript checks in the frontend for best practices.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the amazing deep learning framework
- Flask team for the web framework
- OpenCV and scikit-image teams for image processing capabilities

## 📝 Legacy HTML UI (index.html)

- The file `index.html` is provided as a backup, legacy UI for users who do not wish to use the Next.js frontend.
- It no longer supports real-time progress updates (Socket.IO) and only interacts with the backend via REST API endpoints.
- For the best experience, use the Next.js frontend in the `frontend/` directory.

---

Made with Passion by Dhairya
