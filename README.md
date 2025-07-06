# 🎨 Wallyzer

A powerful and intelligent wallpaper management system that helps you organize, analyze, and optimize your wallpaper collection using advanced AI and computer vision techniques.

![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Next.js](https://img.shields.io/badge/Next.js-14+-black)
![React](https://img.shields.io/badge/React-18+-blue)

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
- 📈 **Image Clustering**: Automatic grouping of similar wallpapers using K-means clustering
- 🔄 **Recursive Analysis**: Option to analyze subdirectories
- ⚡ **Performance Optimization**: Automatic device selection (CPU/CUDA/MPS) and batch size adjustment

## 🛠️ Technical Stack

- **Backend**: Python 3.8+
- **Web Framework**: Flask with SocketIO
- **AI/ML**: PyTorch, torchvision
- **Image Processing**: OpenCV, Pillow, scikit-image
- **Data Processing**: NumPy, SciPy, scikit-learn
- **Real-time Communication**: Flask-SocketIO, eventlet
- **Image Analysis**: ResNet50, Perceptual Hashing
- **Database**: SQLite for result caching
- **Frontend**: Next.js, React, Tailwind CSS, shadcn/ui

## 📋 Requirements

- Python 3.8 or higher
- Node.js 18+ (for frontend)
- CUDA-capable GPU (optional, for faster processing)
- MPS support for Apple Silicon (optional)

## 🚀 Installation

### Backend

1. Clone the repository:

```bash
git clone https://github.com/Dhairya3391/wallyzer.git
cd wallyzer
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

4. (Optional) Run a security audit:

```bash
pip install pip-audit
pip-audit
```

### Frontend

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
pnpm install
```

3. Start the development server:

```bash
pnpm dev
```

## 💻 Usage

### Quick Start (Recommended)

```bash
# Install all dependencies
npm run install:all

# Start both backend and frontend
npm run dev
```

This will start:

- Backend on [http://localhost:8000](http://localhost:8000)
- Frontend on [http://localhost:3000](http://localhost:3000)

### Manual Start

#### Start the Backend

```bash
python wallyzer.py
```

The backend will start on [http://localhost:8000](http://localhost:8000)

#### Start the Frontend

```bash
cd frontend
pnpm dev
```

The frontend will start on [http://localhost:3000](http://localhost:3000)

### Analyze Wallpapers

1. Open your browser and navigate to [http://localhost:3000](http://localhost:3000)
2. Select a directory containing your wallpapers and start the analysis!

## ⚙️ Configuration

The application can be configured through the `Config` class in `wallyzer.py`:

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

You can also use environment variables for deployment flexibility (recommended for production). Copy `env.example` to `.env` and modify the values as needed.

## 🧵 Concurrency Model

- The backend uses **eventlet** (green threads) for async I/O and concurrency, which is optimal for I/O-bound workloads and works seamlessly with Flask-SocketIO.
- For CPU-bound tasks (like image processing), a **ThreadPoolExecutor** is used for parallelism. This hybrid approach provides robust performance for most workloads.
- If you require pure CPU-bound scaling, consider using multiprocessing, though eventlet is recommended for this web+I/O use case.

## 🗂️ Project Structure

```
wallyzer/
├── wallyzer.py              # Backend server and analysis logic
├── requirements.txt         # Python dependencies
├── package.json             # Root package.json with scripts
├── env.example              # Environment variables template
├── LICENSE                  # MIT License
├── CONTRIBUTING.md          # Contribution guidelines
├── CHANGELOG.md             # Project changelog
├── frontend/               # Next.js/React frontend
│   ├── app/                # Main app pages
│   ├── components/         # React components (shadcn/ui, custom)
│   ├── hooks/              # Custom React hooks
│   ├── lib/                # Utilities
│   └── ...
├── analysis_cache.db       # SQLite cache
├── analyzed.log            # Log file
├── README.md               # This file
└── ...
```

## 🧪 Testing & Security

- Run `pip-audit` to check for Python dependency vulnerabilities
- Use `pnpm audit` in the frontend directory for JavaScript dependency security
- Enable ESLint and TypeScript checks in the frontend for code quality

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📋 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.

## 🙏 Acknowledgments

- PyTorch team for the exceptional deep learning framework
- Flask team for the robust web framework
- OpenCV and scikit-image teams for comprehensive image processing capabilities
- Next.js team for the powerful React framework
- shadcn/ui team for the beautiful component library

## 📝 Legacy HTML UI

The `index.html` file is provided as a legacy UI for users who prefer not to use the Next.js frontend. Note that it no longer supports real-time progress updates (Socket.IO) and only interacts with the backend via REST API endpoints. For the optimal experience, we recommend using the Next.js frontend in the `frontend/` directory.

---

Built with ❤️ by Dhairya
