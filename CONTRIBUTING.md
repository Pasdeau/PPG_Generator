# Contributing to PPG_Generator

Thank you for considering contributing to PPG_Generator! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- **Clear title**: Brief description of the issue
- **Steps to reproduce**: Minimal code example
- **Expected vs actual behavior**
- **Environment**:
  - Python version
  - OS (Windows/macOS/Linux)
  - GPU/CPU

Example:
```
Title: Model fails on signals <4 seconds

Python 3.9, Ubuntu 20.04, GPU: RTX 3080

Code:
    ppg = gen_PPG(RR=[800, 800], pulse_type=1, Fd=1000)
    model.predict(ppg)  # IndexError: ...

Expected: Graceful handling or error message
Actual: Crashes with IndexError
```

---

### Suggesting Features

Open an issue with the **Enhancement** label:
- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches considered

---

### Pull Requests

1. **Fork** the repository
2. **Create a branch**: `feature/your-feature-name` or `fix/bug-description`
3. **Make changes** with clear, atomic commits
4. **Test** your changes thoroughly
5. **Submit PR** with description linking to related issues

#### PR Checklist
- [ ] Code follows PEP 8 style guide
- [ ] All tests pass (`pytest`)
- [ ] New features include tests
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated under `[Unreleased]`

---

## 📝 Coding Standards

### Python Style
Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/):
```python
# Good
def generate_ppg_signal(rr_intervals, pulse_type=1, sample_rate=1000):
    """
    Generate synthetic PPG signal.
    
    Args:
        rr_intervals: RR intervals in milliseconds
        pulse_type: Pulse morphology (1-5)
        sample_rate: Sampling frequency in Hz
    
    Returns:
        ppg: Generated signal
    """
    pass

# Bad
def gen(RR,t=1,Fd=1000):pass  # No docstring, unclear names
```

### Documentation
- All public functions **must** have docstrings
- Use Google-style docstrings
- Include:
  - Brief description
  - Args (with types)
  - Returns
  - Example usage (if non-trivial)

### Testing
We use `pytest`. Test files live in `tests/`:
```python
# tests/test_generator.py
def test_ppg_length():
    """Test generated signal has correct length"""
    RR = [800, 800, 800]
    ppg, _, _ = gen_PPG(RR, pulse_type=1, Fd=1000)
    expected_length = sum(RR)  # Approximate
    assert len(ppg) > 0.9 * expected_length
```

Run tests:
```bash
pytest tests/ -v
```

---

## 🌿 Branch Naming

- `feature/add-ecg-fusion`: New features
- `fix/cwt-nan-values`: Bug fixes
- `docs/update-api-reference`: Documentation only
- `refactor/simplify-dataset`: Code refactoring

---

## 📦 Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code restructure (no functional change)
- `test`: Adding tests
- `chore`: Build, dependencies

**Examples**:
```
feat(model): add ONNX export for v4.0 Dual-Stream

- Implement torch.onnx.export() wrapper
- Add validation script for ONNX inference
- Update README with deployment instructions

Closes #42
```

```
fix(dataset): handle CWT edge case for short signals

Previously crashed on signals <2 seconds. Now pads with zeros.

Fixes #38
```

---

## 🧪 Development Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/PPG_Generator.git
cd PPG_Generator

# Create development environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

---

## 🎯 Areas for Contribution

We especially welcome contributions in:

1. **Clinical Validation**:
   - Real-world PPG dataset integration
   - Comparison with hospital-grade monitors

2. **Model Optimization**:
   - Quantization (INT8/FP16)
   - Pruning for edge devices
   - ONNX/TensorRT conversion

3. **Multi-modal Fusion**:
   - PPG + ECG synchronization
   - Accelerometer-based artifact detection

4. **Documentation**:
   - Tutorials and examples
   - API reference improvements
   - Translation to other languages

5. **Testing**:
   - Edge case coverage
   - Performance benchmarks

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the **GNU GPL v3.0**.

---

## 💬 Community

- **GitHub Discussions**: Ask questions, share use cases
- **Email**: wenzheng.wang@lip6.fr (for major proposals)

---

**Thank you for helping make PPG_Generator better!** 🙏
