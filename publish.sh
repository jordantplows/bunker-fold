#!/bin/bash
# Quick publish script for bunker-fold
# Usage: ./publish.sh [testpypi|pypi]

set -e

TARGET=${1:-testpypi}

echo "🔧 Publishing bunker-fold to $TARGET"

# Check if in venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Activating venv..."
    source venv/bin/activate
fi

# Install/update build tools
echo "📦 Installing build tools..."
pip install --upgrade build twine > /dev/null

# Clean old builds
echo "🧹 Cleaning old builds..."
rm -rf dist/ build/ src/*.egg-info

# Build the package
echo "🔨 Building package..."
python -m build

# Check the build
echo "✅ Checking package..."
twine check dist/*

# Show what will be uploaded
echo ""
echo "📋 Files to upload:"
ls -lh dist/
echo ""

# Confirm before upload
read -p "Continue with upload to $TARGET? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Upload cancelled"
    exit 1
fi

# Upload
if [[ "$TARGET" == "testpypi" ]]; then
    echo "🚀 Uploading to TestPyPI..."
    twine upload --repository testpypi dist/*
    echo ""
    echo "✅ Uploaded to TestPyPI!"
    echo ""
    echo "📥 To test install:"
    echo "  pip install --index-url https://test.pypi.org/simple/ \\"
    echo "              --extra-index-url https://pypi.org/simple/ \\"
    echo "              bunker-fold"
    echo ""
    echo "🔗 View at: https://test.pypi.org/project/bunker-fold/"
elif [[ "$TARGET" == "pypi" ]]; then
    echo "🚀 Uploading to PyPI..."
    twine upload dist/*
    echo ""
    echo "✅ Uploaded to PyPI!"
    echo ""
    echo "📥 Install with:"
    echo "  pip install bunker-fold"
    echo ""
    echo "🔗 View at: https://pypi.org/project/bunker-fold/"
else
    echo "❌ Invalid target: $TARGET (use 'testpypi' or 'pypi')"
    exit 1
fi
