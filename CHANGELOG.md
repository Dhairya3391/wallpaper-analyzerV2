# Changelog

All notable changes to Wallyzer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-07-06

### Added

- Initial release of Wallyzer
- AI-powered wallpaper analysis with duplicate detection
- Aesthetic scoring using ResNet models
- Image clustering with K-means algorithm
- Real-time web interface with Next.js
- Support for CPU, CUDA, and Apple Silicon (MPS)
- SQLite caching for analysis results
- Comprehensive project documentation
- Professional project structure with proper configuration files

### Features

- Smart duplicate detection using perceptual hashing and deep learning
- Aesthetic scoring with AI models
- Image clustering and organization
- Real-time progress updates via WebSocket
- Responsive web UI with modern design
- Batch processing for large image collections
- Configurable analysis parameters
- Cross-platform support (Windows, macOS, Linux)

### Technical Stack

- Backend: Python 3.8+, Flask, Flask-SocketIO
- AI/ML: PyTorch, torchvision, scikit-learn
- Image Processing: OpenCV, Pillow, scikit-image
- Frontend: Next.js, React, TypeScript, Tailwind CSS, shadcn/ui
- Database: SQLite for caching
- Real-time Communication: WebSockets

---

## [Unreleased]

### Planned

- Cloud storage integration (Google Drive)
- User authentication and multi-user support
- Additional export formats (CSV, JSON, ZIP)
- Enhanced clustering with transformer models
- Video thumbnail analysis
- Performance optimizations
- Additional AI models for better analysis
