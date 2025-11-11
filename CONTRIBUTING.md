# Contributing to PyBanglaCodeActAgent

Thank you for your interest in contributing to PyBanglaCodeActAgent! This project aims to advance code generation capabilities for Bangla (Bengali) programming tasks.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - System information (OS, Python version, GPU specs)
   - Relevant logs or error messages

### Contributing Code

1. **Fork the repository** to your GitHub account
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/PyBanglaCodeActAgent.git
   cd PyBanglaCodeActAgent
   ```

3. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Set up your development environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Make your changes**:
   - Write clean, readable code
   - Follow PEP 8 style guidelines
   - Add docstrings for new functions/classes
   - Add comments for complex logic
   - Update documentation if needed

6. **Test your changes**:
   - Ensure existing tests still pass
   - Add tests for new functionality
   - Test with sample data from `dev.csv`

7. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
   
   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `refactor:` for code refactoring
   - `test:` for adding tests
   - `chore:` for maintenance tasks

8. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

9. **Create a Pull Request**:
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Provide a clear description of your changes
   - Link related issues

### Code Style Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use meaningful variable and function names
- Keep functions focused and modular
- Add type hints where appropriate
- Document complex algorithms and design decisions

### Testing Guidelines

- Test with different models if possible
- Verify behavior with edge cases
- Check memory usage for large datasets
- Ensure compatibility with different Python versions (3.9+)

### Documentation

When adding new features:
- Update the README.md if user-facing changes
- Add docstrings to new functions/classes
- Update inline comments for complex logic
- Consider adding examples in notebooks

## Development Setup

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for model inference)
- Git

### Running Tests

```bash
python test.py
```

### Running the Agent

```bash
python PyBanglaCodeAct.py --input dev.csv --output submission.json
```

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others when you can
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md) (if available)

## Questions?

- Open an issue for general questions
- Tag maintainers for urgent matters
- Check existing documentation first

## Recognition

Contributors will be acknowledged in:
- The project README
- Release notes
- Academic papers (for significant contributions)

Thank you for helping make PyBanglaCodeAct better! 🎉
