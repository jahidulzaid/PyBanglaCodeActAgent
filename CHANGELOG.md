# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-11

### Added
- Initial release of PyBanglaCodeActAgent
- BanglaCodeAct agent implementation with Thought-Code-Observation loop
- Support for Bangla programming instruction processing
- Sandboxed Python REPL with timeout protection
- Safe retry logic with configurable attempts
- Command-line interface with argparse
- Comprehensive logging with color-coded output
- Syntax highlighting for code blocks
- File format validation for submissions
- Example usage script
- Configuration file for easy customization
- Unit tests and scoring utilities
- Sample notebooks for prompting and fine-tuning
- Complete documentation (README, CONTRIBUTING, CODE_OF_CONDUCT)
- MIT License
- Requirements file with all dependencies
- Setup.py for package installation
- GitHub Actions workflow for CI/CD

### Features
- Achieves 94.0% pass@1 on mHumanEval Bangla dev set
- Support for multiple LLM models (Qwen, DeepSeek, Llama, etc.)
- Self-consistency with majority voting
- Iterative self-correction with execution feedback
- Efficient inference with vLLM
- Tensor parallelism support for multi-GPU setups

### Documentation
- Professional README with installation and usage instructions
- Contributing guidelines
- Code of conduct
- Example scripts and notebooks
- Inline code documentation with docstrings

## [Unreleased]

### Planned
- Add more language support
- Implement pass@k evaluation
- Add fine-tuning examples
- Optimize memory usage
- Add more test cases
- Create Docker support
- Add web interface
