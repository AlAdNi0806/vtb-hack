#!/bin/bash

# Real-time Conversational AI Setup Script
# This script sets up the development environment for the conversational AI application

set -e  # Exit on any error

echo "🚀 Setting up Real-time Conversational AI Application"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    print_status "Checking operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_success "Linux detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_success "macOS detected"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        print_success "Windows detected"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js $NODE_VERSION found"
    else
        print_error "Node.js is required but not installed"
        exit 1
    fi
    
    # Check npm
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version)
        print_success "npm $NPM_VERSION found"
    else
        print_error "npm is required but not installed"
        exit 1
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | cut -d' ' -f3)
        print_success "Git $GIT_VERSION found"
    else
        print_error "Git is required but not installed"
        exit 1
    fi
}

# Setup backend
setup_backend() {
    print_status "Setting up backend..."
    
    cd backend
    
    # Create virtual environment
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    if [[ "$OS" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Create environment file
    if [ ! -f .env ]; then
        print_status "Creating backend environment file..."
        cp .env.example .env
        print_warning "Please edit backend/.env and add your Cerebras API key"
    else
        print_success "Backend environment file already exists"
    fi
    
    # Run tests
    print_status "Running backend tests..."
    python run_tests.py
    
    cd ..
    print_success "Backend setup completed"
}

# Setup frontend
setup_frontend() {
    print_status "Setting up frontend..."
    
    cd frontend
    
    # Install dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    # Create environment file
    if [ ! -f .env.local ]; then
        print_status "Creating frontend environment file..."
        cp .env.example .env.local
        print_success "Frontend environment file created"
    else
        print_success "Frontend environment file already exists"
    fi
    
    # Build the application
    print_status "Building frontend application..."
    npm run build
    
    cd ..
    print_success "Frontend setup completed"
}

# Create startup scripts
create_startup_scripts() {
    print_status "Creating startup scripts..."
    
    # Backend startup script
    cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "Starting Conversational AI Backend..."
cd backend
source venv/bin/activate
python main.py
EOF
    
    # Frontend startup script
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "Starting Conversational AI Frontend..."
cd frontend
npm run dev
EOF
    
    # Make scripts executable
    chmod +x start_backend.sh start_frontend.sh
    
    # Windows batch files
    cat > start_backend.bat << 'EOF'
@echo off
echo Starting Conversational AI Backend...
cd backend
call venv\Scripts\activate
python main.py
EOF
    
    cat > start_frontend.bat << 'EOF'
@echo off
echo Starting Conversational AI Frontend...
cd frontend
npm run dev
EOF
    
    print_success "Startup scripts created"
}

# Display final instructions
show_final_instructions() {
    echo ""
    echo "🎉 Setup completed successfully!"
    echo "================================"
    echo ""
    echo "Next steps:"
    echo "1. Edit backend/.env and add your Cerebras API key:"
    echo "   CEREBRAS_API_KEY=csk-6kjk8yewdwdhkc8ndj5686rcj2te3tfyyr6dw669knd3wy33"
    echo ""
    echo "2. Start the backend (in one terminal):"
    if [[ "$OS" == "windows" ]]; then
        echo "   ./start_backend.bat"
    else
        echo "   ./start_backend.sh"
    fi
    echo ""
    echo "3. Start the frontend (in another terminal):"
    if [[ "$OS" == "windows" ]]; then
        echo "   ./start_frontend.bat"
    else
        echo "   ./start_frontend.sh"
    fi
    echo ""
    echo "4. Open your browser and go to: http://localhost:3000"
    echo ""
    echo "📚 Documentation:"
    echo "   - Quick Start: docs/quick-start.md"
    echo "   - API Reference: docs/api-reference.md"
    echo "   - Deployment: docs/deployment.md"
    echo "   - Architecture: docs/architecture.md"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   - Check that all prerequisites are installed"
    echo "   - Ensure ports 3000 and 8000 are available"
    echo "   - Verify your Cerebras API key is correct"
    echo "   - Check browser console for any errors"
    echo ""
    echo "Happy conversing with AI! 🤖"
}

# Main execution
main() {
    check_os
    check_prerequisites
    setup_backend
    setup_frontend
    create_startup_scripts
    show_final_instructions
}

# Run main function
main
