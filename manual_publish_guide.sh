#!/bin/bash
# Interactive guide for first-time PyPI publishing
# Run: bash manual_publish_guide.sh

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Bunker-Fold PyPI Publishing Guide                        ║"
echo "║  First-time manual upload walkthrough                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check prerequisites
echo "📋 Step 1: Checking prerequisites..."
echo ""

if [[ ! -d "venv" ]]; then
    echo "❌ No venv found. Creating one..."
    python3 -m venv venv
fi

source venv/bin/activate

if ! command -v twine &> /dev/null; then
    echo "📦 Installing build tools..."
    pip install build twine
fi

echo "✅ Prerequisites OK"
echo ""

# Step 2: Get API tokens
echo "═══════════════════════════════════════════════════════════"
echo "📝 Step 2: Get your PyPI API tokens"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "You need to create API tokens for both TestPyPI and PyPI."
echo ""
echo "1. TestPyPI (for testing):"
echo "   → Open: https://test.pypi.org/manage/account/token/"
echo "   → Click 'Add API token'"
echo "   → Name: bunker-fold-test"
echo "   → Scope: Entire account"
echo "   → Click 'Add token' and COPY IT"
echo ""

read -p "Press Enter after you have your TestPyPI token ready..."
echo ""

read -sp "Paste your TestPyPI token: " TEST_TOKEN
echo ""

if [[ ! "$TEST_TOKEN" =~ ^pypi- ]]; then
    echo "⚠️  Warning: Token should start with 'pypi-'"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "2. PyPI (for production - get this after testing):"
echo "   → Open: https://pypi.org/manage/account/token/"
echo "   → Same steps as TestPyPI"
echo ""

read -p "Do you have your PyPI token ready? (y/N) " -n 1 -r
echo
HAVE_PROD_TOKEN=$REPLY

PROD_TOKEN=""
if [[ $HAVE_PROD_TOKEN =~ ^[Yy]$ ]]; then
    read -sp "Paste your PyPI token: " PROD_TOKEN
    echo ""
fi

# Step 3: Create .pypirc
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "🔧 Step 3: Creating ~/.pypirc config"
echo "═══════════════════════════════════════════════════════════"
echo ""

if [[ -f ~/.pypirc ]]; then
    echo "⚠️  ~/.pypirc already exists"
    read -p "Backup and overwrite? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp ~/.pypirc ~/.pypirc.backup.$(date +%s)
        echo "✅ Backed up to ~/.pypirc.backup.*"
    else
        echo "Skipping .pypirc creation"
        SKIP_PYPIRC=1
    fi
fi

if [[ -z "$SKIP_PYPIRC" ]]; then
    cat > ~/.pypirc << EOF
[testpypi]
  username = __token__
  password = $TEST_TOKEN

[pypi]
  username = __token__
  password = ${PROD_TOKEN:-YOUR_PYPI_TOKEN_HERE}
EOF
    chmod 600 ~/.pypirc
    echo "✅ Created ~/.pypirc with your tokens"
fi

# Step 4: Build
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "🔨 Step 4: Building the package"
echo "═══════════════════════════════════════════════════════════"
echo ""

rm -rf dist/ build/ src/*.egg-info

python -m build

echo ""
echo "✅ Built:"
ls -lh dist/

echo ""
twine check dist/*

# Step 5: Upload to TestPyPI
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "🚀 Step 5: Uploading to TestPyPI"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "This will upload bunker-fold v0.1.0 to TEST PyPI"
echo ""

read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Upload cancelled. You can upload manually later with:"
    echo "   twine upload --repository testpypi dist/*"
    exit 0
fi

echo ""
echo "Uploading to TestPyPI..."
twine upload --repository testpypi dist/*

echo ""
echo "✅ Upload complete!"
echo ""
echo "🔗 View at: https://test.pypi.org/project/bunker-fold/"

# Step 6: Test installation
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "🧪 Step 6: Test installation"
echo "═══════════════════════════════════════════════════════════"
echo ""

read -p "Test installation now? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo "Creating test environment..."
    python -m venv test_bunker_install
    source test_bunker_install/bin/activate

    echo "Installing from TestPyPI..."
    pip install --index-url https://test.pypi.org/simple/ \
                --extra-index-url https://pypi.org/simple/ \
                bunker-fold

    echo ""
    echo "Testing CLI..."
    bunker --help

    echo ""
    echo "Testing Python import..."
    python -c "import bunker; print(f'✅ bunker v{bunker.__version__}')"

    echo ""
    echo "✅ Installation test passed!"

    # Cleanup
    deactivate
    rm -rf test_bunker_install
    source venv/bin/activate
fi

# Step 7: Production upload
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "🎯 Step 7: Upload to production PyPI"
echo "═══════════════════════════════════════════════════════════"
echo ""

if [[ -z "$PROD_TOKEN" ]] || [[ "$PROD_TOKEN" == "YOUR_PYPI_TOKEN_HERE" ]]; then
    echo "⚠️  You haven't set up a PyPI production token yet."
    echo ""
    echo "Before uploading to production:"
    echo "1. Get token: https://pypi.org/manage/account/token/"
    echo "2. Update ~/.pypirc with the token"
    echo "3. Run: twine upload dist/*"
    echo ""
    echo "Or run this script again with the token ready."
else
    echo "⚠️  WARNING: This will upload to PRODUCTION PyPI"
    echo "   Once uploaded, this version CANNOT be deleted!"
    echo ""
    read -p "Upload to production PyPI now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        twine upload dist/*
        echo ""
        echo "🎉 SUCCESS! Package published to PyPI!"
        echo ""
        echo "🔗 View at: https://pypi.org/project/bunker-fold/"
        echo ""
        echo "Install with: pip install bunker-fold"
    else
        echo ""
        echo "Production upload skipped. When ready:"
        echo "   twine upload dist/*"
    fi
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  🎉 Publishing workflow complete!                         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
