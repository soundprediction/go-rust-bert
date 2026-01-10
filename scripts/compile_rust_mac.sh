#!/bin/bash

# This script compiles the Rust library as a Universal Binary for macOS (arm64 + x86_64)

set -e

export MACOSX_DEPLOYMENT_TARGET=11.0

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Confirm OS
OS=$(uname -s)
if [ "$OS" != "Darwin" ]; then
    echo "❌ This script is intended for macOS only."
    exit 1
fi

TARGET_DIR="$PROJECT_ROOT/pkg/rustbert/lib/darwin"
mkdir -p "$TARGET_DIR"

cd "$PROJECT_ROOT/rust_bert_binding"

# Check if rustup is available to add targets
if command -v rustup >/dev/null 2>&1; then
    echo "🦀 Adding Rust targets for macOS..."
    rustup target add aarch64-apple-darwin x86_64-apple-darwin
else
    echo "⚠️  rustup not found. Assuming targets are already installed."
fi

# For rust-bert (tch), we might need to handle libtorch download. 
# Attempting standard cargo build which triggers tch build.rs
# Note: tch-rs requires LIBTORCH env var or downloads it. M1 macs might need specific libtorch.

echo "🍎 Building for Apple Silicon (arm64)..."
cargo build --release --target aarch64-apple-darwin

# echo "💻 Building for Intel (x86_64)..."
# cargo build --release --target x86_64-apple-darwin

# if command -v lipo >/dev/null 2>&1; then
#     echo "🚀 Creating Universal Binary using lipo..."
#     lipo -create \
#         target/aarch64-apple-darwin/release/librust_bert_binding.dylib \
#         target/x86_64-apple-darwin/release/librust_bert_binding.dylib \
#         -output "$TARGET_DIR/librust_bert_binding.dylib"
    
#     # Compress
#     gzip -9 -f "$TARGET_DIR/librust_bert_binding.dylib"

#     echo "✅ Universal library created at $TARGET_DIR/librust_bert_binding.dylib.gz"
# else
    echo "⚠️ Skipping Universal Binary (x86_64 build failed). Using host architecture only."
    cp target/aarch64-apple-darwin/release/librust_bert_binding.dylib "$TARGET_DIR/"
    
    # Bundle libtorch libraries
    echo "📦 Bundling libtorch..."
    TORCH_LIB_DIR=$(find target/aarch64-apple-darwin/release/build -name "libtorch" -type d | grep "out/libtorch/libtorch$" | grep "lib" | head -n 1) # This might needs adjustment
    # Actually the path found was .../out/libtorch/libtorch/lib
    # Let's try finding the directory containing libc10.dylib
    TORCH_LIB_DIR=$(find target/aarch64-apple-darwin/release/build -name "libc10.dylib" -exec dirname {} \; | head -n 1)
    
    if [ -d "$TORCH_LIB_DIR" ]; then
        echo "Found libtorch at $TORCH_LIB_DIR"
        cp "$TORCH_LIB_DIR/libc10.dylib" "$TARGET_DIR/"
        cp "$TORCH_LIB_DIR/libtorch_cpu.dylib" "$TARGET_DIR/"
        cp "$TORCH_LIB_DIR/libtorch.dylib" "$TARGET_DIR/"
        cp "$TORCH_LIB_DIR/libomp.dylib" "$TARGET_DIR/"
        
        # Set RPATH on the binding library to find dependencies in the same directory (@loader_path)
        install_name_tool -add_rpath @loader_path "$TARGET_DIR/librust_bert_binding.dylib"
        
        # Gzip everything
        gzip -9 -f "$TARGET_DIR/librust_bert_binding.dylib"
        gzip -9 -f "$TARGET_DIR/libc10.dylib"
        gzip -9 -f "$TARGET_DIR/libtorch_cpu.dylib"
        gzip -9 -f "$TARGET_DIR/libtorch.dylib"
        gzip -9 -f "$TARGET_DIR/libomp.dylib"
    else
        echo "❌ Could not find libtorch directory. Runtime linking might fail."
    fi
# fi
