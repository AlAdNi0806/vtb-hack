# Deployment Guide for Ubuntu Server

This guide provides step-by-step instructions for deploying the Real-time Conversational AI application on an Ubuntu server.

## Prerequisites

### System Requirements
- Ubuntu 20.04 LTS or later
- Minimum 8GB RAM (16GB recommended)
- 50GB+ available disk space
- GPU support (NVIDIA GPU with CUDA recommended for optimal performance)
- Internet connection for downloading models and dependencies

### Required Software
- Python 3.9+
- Node.js 18+
- Git
- NVIDIA drivers (if using GPU)

## Initial Server Setup

### 1. Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install System Dependencies
```bash
# Essential build tools
sudo apt install -y build-essential curl wget git

# Python development tools
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Audio processing libraries
sudo apt install -y portaudio19-dev libasound2-dev libsndfile1-dev

# Node.js (using NodeSource repository)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installations
python3 --version
node --version
npm --version
```

### 3. Install NVIDIA Drivers and CUDA (Optional but Recommended)
```bash
# Check if NVIDIA GPU is available
lspci | grep -i nvidia

# Install NVIDIA drivers
sudo apt install -y nvidia-driver-470

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cuda

# Reboot to load drivers
sudo reboot
```

## Application Deployment

### 1. Clone Repository
```bash
cd /opt
sudo git clone <your-repository-url> conversational-ai
sudo chown -R $USER:$USER /opt/conversational-ai
cd /opt/conversational-ai
```

### 2. Backend Setup

#### Create Python Virtual Environment
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

#### Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch with CUDA support (if GPU available)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit environment file
nano .env
```

Update the `.env` file with your configuration:
```env
CEREBRAS_API_KEY=your_actual_api_key_here
HOST=0.0.0.0
PORT=8000
DEBUG=False
LOG_LEVEL=INFO
```

#### Test Backend
```bash
# Run tests
python run_tests.py

# Start backend (test)
python main.py
```

### 3. Frontend Setup

#### Install Dependencies
```bash
cd ../frontend
npm install
```

#### Configure Environment
```bash
# Copy environment template
cp .env.example .env.local

# Edit environment file
nano .env.local
```

Update the `.env.local` file:
```env
NEXT_PUBLIC_API_URL=http://your-server-ip:8000
NEXT_PUBLIC_WS_URL=ws://your-server-ip:8000/ws
NEXT_PUBLIC_DEBUG=false
```

#### Build Frontend
```bash
npm run build
```

## Production Deployment

### 1. Create System Services

#### Backend Service
```bash
sudo nano /etc/systemd/system/conversational-ai-backend.service
```

Add the following content:
```ini
[Unit]
Description=Conversational AI Backend
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/conversational-ai/backend
Environment=PATH=/opt/conversational-ai/backend/venv/bin
ExecStart=/opt/conversational-ai/backend/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Frontend Service
```bash
sudo nano /etc/systemd/system/conversational-ai-frontend.service
```

Add the following content:
```ini
[Unit]
Description=Conversational AI Frontend
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/conversational-ai/frontend
Environment=PATH=/usr/bin:/bin
ExecStart=/usr/bin/npm start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 2. Enable and Start Services
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable conversational-ai-backend
sudo systemctl enable conversational-ai-frontend

# Start services
sudo systemctl start conversational-ai-backend
sudo systemctl start conversational-ai-frontend

# Check status
sudo systemctl status conversational-ai-backend
sudo systemctl status conversational-ai-frontend
```

### 3. Configure Nginx (Reverse Proxy)

#### Install Nginx
```bash
sudo apt install -y nginx
```

#### Configure Nginx
```bash
sudo nano /etc/nginx/sites-available/conversational-ai
```

Add the following configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket
    location /ws {
        proxy_pass http://localhost:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Enable Nginx Configuration
```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/conversational-ai /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### 4. Configure Firewall
```bash
# Enable UFW
sudo ufw enable

# Allow SSH
sudo ufw allow ssh

# Allow HTTP and HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Allow backend port (if direct access needed)
sudo ufw allow 8000

# Check status
sudo ufw status
```

## SSL/HTTPS Setup (Optional but Recommended)

### Using Let's Encrypt
```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Test automatic renewal
sudo certbot renew --dry-run
```

## Monitoring and Maintenance

### 1. Log Management
```bash
# View backend logs
sudo journalctl -u conversational-ai-backend -f

# View frontend logs
sudo journalctl -u conversational-ai-frontend -f

# View Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 2. System Monitoring
```bash
# Check system resources
htop
df -h
free -h

# Check GPU usage (if available)
nvidia-smi
```

### 3. Application Health Checks
```bash
# Check backend health
curl http://localhost:8000/health

# Check frontend
curl http://localhost:3000

# Check services status
sudo systemctl status conversational-ai-backend conversational-ai-frontend nginx
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   sudo lsof -i :8000
   sudo lsof -i :3000
   ```

2. **Permission Issues**
   ```bash
   sudo chown -R $USER:$USER /opt/conversational-ai
   ```

3. **Memory Issues**
   - Increase swap space
   - Monitor memory usage with `htop`
   - Consider upgrading server resources

4. **GPU Issues**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Performance Optimization

1. **Enable GPU acceleration** for AI models
2. **Configure model caching** to reduce load times
3. **Use process managers** like PM2 for Node.js
4. **Implement load balancing** for multiple instances
5. **Configure CDN** for static assets

## Security Considerations

1. **Keep system updated**
2. **Use strong passwords and SSH keys**
3. **Configure fail2ban** for intrusion prevention
4. **Regular security audits**
5. **Monitor access logs**
6. **Use HTTPS in production**

## Backup and Recovery

1. **Database backups** (if applicable)
2. **Configuration backups**
3. **Model checkpoints**
4. **Regular system snapshots**

For additional support, refer to the project documentation or contact the development team.
