#!/bin/bash
# Compile HumpDay JSS Paper
# Usage: ./compile_paper.sh

set -e  # Exit on error

echo "🔧 Setting up JSS paper compilation..."

# Check if Tectonic is installed
if ! command -v tectonic &> /dev/null; then
    echo "❌ Tectonic not found. Install with:"
    echo "   macOS: brew install tectonic"
    echo "   Ubuntu: cargo install tectonic"
    echo "   Or: pip install tectonic (if available)"
    exit 1
fi

# Download JSS template if not present
if [ ! -f "jss.cls" ]; then
    echo "📥 Downloading JSS template..."
    curl -O https://www.jstatsoft.org/public/journals/1/jss.cls
    curl -O https://www.jstatsoft.org/public/journals/1/jsslogo.jpg
fi

# Compile paper
echo "📝 Compiling HumpDay JSS paper with Tectonic..."

# Tectonic handles multiple passes automatically
tectonic humpday-jss.tex

# Cleanup auxiliary files
echo "🧹 Cleaning up auxiliary files..."
rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lot *.lof

echo "✅ Paper compiled successfully: humpday-jss.pdf"

# Open PDF if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    open humpday-jss.pdf
fi