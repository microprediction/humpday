# HumpDay JSS Paper

Journal of Statistical Software submission for the HumpDay optimization comparison platform.

## 📋 Setup

### 1. Install paper dependencies:
```bash
# From project root
uv sync --extra paper
```

### 2. Install Tectonic (modern LaTeX):
```bash
# macOS
brew install tectonic

# Linux (with Rust)
cargo install tectonic

# Or system package manager
sudo apt install tectonic  # Ubuntu 22.04+
```

## 🔨 Compilation

### Quick compilation:
```bash
cd paper/
./compile_paper.sh
```

### Manual compilation:
```bash
cd paper/

# Download JSS template
curl -O https://www.jstatsoft.org/public/journals/1/jss.cls

# Compile
pdflatex humpday-jss.tex
pdflatex humpday-jss.tex  # Second pass for references
```

## 📁 Structure

```
paper/
├── compile_paper.sh     # Automated compilation
├── humpday-jss.tex      # Main LaTeX document
├── humpday-jss.bib      # Bibliography 
├── figures/             # Generated plots
├── data/                # Experimental results
└── README.md           # This file
```

## 📊 Generating Results

Run experiments and generate figures:
```bash
# From project root
uv run python experiments/benchmark_comparison.py
uv run python experiments/generate_paper_figures.py
```

## ✅ Submission Checklist

- [ ] All figures generated and placed in `figures/`
- [ ] Experimental results documented in `data/`  
- [ ] Bibliography complete in `humpday-jss.bib`
- [ ] Code availability section updated
- [ ] Abstract under 150 words
- [ ] Keywords selected (5-8)
- [ ] Reproducible examples included
- [ ] PDF compiles without errors

## 🎯 JSS Requirements

- **Open source**: ✅ MIT license
- **Reproducible**: ✅ Browser demo + code
- **Cross-platform**: ✅ Browser-based
- **Documentation**: ✅ Interactive tutorials
- **Software availability**: ✅ GitHub + live demo