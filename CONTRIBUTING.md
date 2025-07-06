# Contributing to Wallyzer

Thank you for your interest in contributing to Wallyzer! This document provides guidelines and information for contributors.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up the development environment**:

   ```bash
   # Install Python dependencies
   pip install -r requirements.txt

   # Install Node.js dependencies
   cd frontend
   pnpm install
   ```

## 🛠️ Development Setup

### Backend (Python)

- Python 3.8 or higher
- Install dependencies: `pip install -r requirements.txt`
- Run the server: `python wallyzer.py`

### Frontend (Next.js)

- Node.js 18 or higher
- Install dependencies: `cd frontend && pnpm install`
- Run the development server: `pnpm dev`

### Running Both

```bash
# From the root directory
npm run dev
```

## 📝 Code Style Guidelines

### Python

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings for functions and classes
- Keep functions focused and single-purpose

### TypeScript/React

- Use TypeScript for all new code
- Follow the existing component patterns
- Use shadcn/ui components when possible
- Prefer functional components with hooks

## 🧪 Testing

### Backend Testing

```bash
# Run Python tests (when implemented)
python -m pytest
```

### Frontend Testing

```bash
cd frontend
pnpm test
```

## 📦 Making Changes

1. **Create a feature branch** from `main`
2. **Make your changes** following the code style guidelines
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Commit your changes** with clear, descriptive messages

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

Examples:

- `feat(ui): add dark mode toggle`
- `fix(backend): resolve memory leak in image processing`
- `docs(readme): update installation instructions`

## 🔍 Pull Request Process

1. **Ensure your code follows the style guidelines**
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Submit a pull request** with a clear description
5. **Wait for review** and address any feedback

## 🐛 Reporting Issues

When reporting issues, please include:

- **Operating system** and version
- **Python version** (for backend issues)
- **Node.js version** (for frontend issues)
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Screenshots** (if applicable)

## 🎯 Areas for Contribution

- **Performance improvements** in image processing
- **New AI models** for better analysis
- **UI/UX enhancements** in the frontend
- **Additional export formats** for results
- **Cloud storage integration**
- **Documentation improvements**
- **Test coverage** expansion

## 📞 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Code Review**: For specific code-related questions

## 📄 License

By contributing to Wallyzer, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Wallyzer! 🎨
