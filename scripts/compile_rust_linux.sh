#!/bin/bash

# This script compiles the Rust library for Linux (amd64)
# It bundles the required LibTorch libraries.

set -e

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

TARGET_DIR="$PROJECT_ROOT/pkg/rustbert/lib/linux-amd64"
mkdir -p "$TARGET_DIR"

cd "$PROJECT_ROOT/rust_bert_binding"

echo "🐧 Building for Linux (amd64)..."
cargo build --release

# Determine paths
LIB_NAME="librust_bert_binding.so"
SOURCE_LIB="target/release/$LIB_NAME"

if [ ! -f "$SOURCE_LIB" ]; then
    echo "❌ Build failed? Could not find $SOURCE_LIB"
    exit 1
fi

cp "$SOURCE_LIB" "$TARGET_DIR/"

# Bundle libtorch libraries
echo "📦 Bundling libtorch..."
# Attempt to find where tch-rs downloaded/built libtorch
# Look for libc10.so in the target/release/build directory
TORCH_LIB_DIR=$(find target/release/build -name "libc10.so" -exec dirname {} \; | head -n 1)

if [ -d "$TORCH_LIB_DIR" ]; then
    echo "Found libtorch at $TORCH_LIB_DIR"
    cp "$TORCH_LIB_DIR/libc10.so" "$TARGET_DIR/"
    cp "$TORCH_LIB_DIR/libtorch_cpu.so" "$TARGET_DIR/"
    cp "$TORCH_LIB_DIR/libtorch.so" "$TARGET_DIR/"
    # Often openmp is needed, on linux it's usually libgomp.so.1 but might be bundled as libomp.so?
    # checking for libgomp
    if [ -f "$TORCH_LIB_DIR/libgomp.so.1" ]; then
        cp "$TORCH_LIB_DIR/libgomp.so.1" "$TARGET_DIR/"
    fi
    
    # Try to set RPATH to $ORIGIN so it finds bundled libs
    if command -v patchelf >/dev/null 2>&1; then
        echo "🔧 Setting RPATH to \$ORIGIN using patchelf..."
        patchelf --set-rpath '$ORIGIN' "$TARGET_DIR/$LIB_NAME"
    else
        echo "⚠️  patchelf not found. You might need to set LD_LIBRARY_PATH to the library directory at runtime."
    fi
    
    # Gzip everything
    gzip -9 -f "$TARGET_DIR/$LIB_NAME"
    gzip -9 -f "$TARGET_DIR/libc10.so"
    gzip -9 -f "$TARGET_DIR/libtorch_cpu.so"
    gzip -9 -f "$TARGET_DIR/libtorch.so"
    if [ -f "$TARGET_DIR/libgomp.so.1" ]; then
       gzip -9 -f "$TARGET_DIR/libgomp.so.1"
    fi
    
    echo "✅ Linux library bundle created at $TARGET_DIR"
else
    echo "❌ Could not find libtorch directory. Runtime linking might fail."
fi
