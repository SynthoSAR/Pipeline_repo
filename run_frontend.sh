#!/bin/bash

# Video Processing Pipeline Frontend Launcher
# This script builds and runs the Rust GUI application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎬 Video Processing Pipeline Frontend${NC}"
echo "=================================================="

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ Cargo (Rust) is not installed${NC}"
    echo -e "${YELLOW}Please install Rust from https://rustup.rs/${NC}"
    exit 1
fi

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: CUDA compiler (nvcc) not found${NC}"
    echo -e "${YELLOW}   Video processing may fail without CUDA support${NC}"
fi

# Check if OpenCV libraries exist
if ! pkg-config --exists opencv4 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Warning: OpenCV development libraries not found${NC}"
    echo -e "${YELLOW}   CUDA compilation may fail${NC}"
fi

echo -e "${BLUE}🔧 Building application...${NC}"

# Build the application in release mode
if cargo build --release; then
    echo -e "${GREEN}✅ Build successful${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo -e "${BLUE}🚀 Starting Video Processing Pipeline Frontend...${NC}"
echo ""

# Run the application
exec ./target/release/pipeline-frontend