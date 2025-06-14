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
- 📊 **Image Clustering**: Automatic grouping of similar wallpapers using K-means clustering
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

## 📋 Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)
- MPS support for Apple Silicon (optional)

## 🚀 Installation

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

## 💻 Usage

1. Start the server:

```bash
python wallpaper_analyzer.py
```

2. Open your web browser and navigate to:

```
http://localhost:8000
```

3. Select a directory containing your wallpapers and start the analysis!

## 🔧 Configuration

The application can be configured through the `Config` class in `wallpaper_analyzer.py`:

- `DEVICE`: Processing device (CPU/CUDA/MPS)
- `BATCH_SIZE`: Number of images processed simultaneously
- `MAX_WORKERS`: Maximum number of parallel workers
- `DEFAULT_SIMILARITY_THRESHOLD`: Threshold for duplicate detection
- `DEFAULT_AESTHETIC_THRESHOLD`: Threshold for aesthetic scoring
- `MIN_CLUSTERS`: Minimum number of clusters for image grouping
- `MAX_CLUSTERS`: Maximum number of clusters for image grouping
- `CLUSTER_FEATURE_DIM`: Feature dimension for clustering
- `HOST`: Server host address
- `PORT`: Server port number
- `DEBUG`: Debug mode flag
- `ENABLE_APP_LOGGING`: Application logging flag
- `ENABLE_SOCKET_LOGGING`: Socket.IO logging flag
- `DB_PATH`: Path to SQLite database for caching

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the amazing deep learning framework
- Flask team for the web framework
- OpenCV and scikit-image teams for image processing capabilities

---

Made with Passion by Dhairya
