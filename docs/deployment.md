# Deployment Guide

This guide provides comprehensive instructions for deploying the StockTracker application in various environments, from local development to production servers and cloud platforms.

## 1. Deployment Overview

### Deployment Options

StockTracker can be deployed in several ways depending on your requirements:

1. **Local Development Deployment**
   - Ideal for development, testing, and personal use
   - Runs directly on your local machine
   - Suitable for individual users and developers

2. **Server Deployment**
   - Deploy on a dedicated Linux server
   - Can be configured as a background service
   - Suitable for team usage or small-scale production

3. **Containerized Deployment**
   - Using Docker for consistent environments
   - Easy to scale and manage
   - Platform-independent deployment

4. **Cloud Deployment**
   - Deploy to major cloud providers (AWS, Azure, Google Cloud)
   - Leverages cloud scalability and reliability
   - Suitable for production environments with high availability requirements

### System Requirements

#### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4 GB
- **Disk Space**: 10 GB free space
- **Operating System**: 
  - Ubuntu 20.04+ (recommended)
  - CentOS 7+ 
  - Debian 10+
  - Windows 10+ (for local development)
  - macOS 10.15+ (for local development)

#### Recommended Requirements
- **CPU**: 4 cores or more
- **RAM**: 8 GB or more
- **Disk Space**: 50 GB free space (for data storage)
- **Operating System**: Ubuntu 22.04 LTS or newer

### Prerequisites

Before deploying StockTracker, ensure the following prerequisites are met:

1. **Python 3.12+** installed on the system
2. **Git** for version control
3. **uv** package manager (recommended) or pip
4. **Docker** (for containerized deployment)
5. **Docker Compose** (for multi-container setups)
6. **Nginx** (for reverse proxy in server deployment)
7. **Systemd** (for service management on Linux)
8. **SSL Certificate** (for HTTPS in production)

For cloud deployments, you'll also need:
- Appropriate cloud provider account with necessary permissions
- Understanding of cloud-specific networking and security groups
## 2. Local Deployment

Local deployment is ideal for development, testing, and personal use of StockTracker. This section covers setting up the application on your local machine.

### Development Environment Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/StockTracker.git
   cd StockTracker
   ```

2. **Install Python 3.12+**
   
   Ensure you have Python 3.12 or newer installed:
   ```bash
   python --version
   ```

3. **Install uv Package Manager**
   
   StockTracker uses `uv` as the recommended package manager:
   ```bash
   pip install uv
   ```

4. **Install Dependencies**
   
   Install all required dependencies using uv:
   ```bash
   uv sync
   ```

5. **Verify Installation**
   
   Test that the installation was successful:
   ```bash
   python main.py --help
   ```

### Running the Web Interface Locally

StockTracker provides a web interface built with Streamlit for interactive use:

1. **Start the Web Interface**
   ```bash
   uv run streamlit run app.py
   ```

2. **Access the Application**
   
   The web interface will be available at `http://localhost:8501` by default.

3. **Customize Streamlit Settings**
   
   You can customize the port and other settings:
   ```bash
   uv run streamlit run app.py --server.port 8502 --server.address 0.0.0.0
   ```

### Background Service Setup

For continuous operation without requiring an active terminal:

1. **Using nohup (Linux/macOS)**
   ```bash
   nohup uv run streamlit run app.py > stocktracker.log 2>&1 &
   ```

2. **Using screen (Linux/macOS)**
   ```bash
   screen -S stocktracker
   uv run streamlit run app.py
   # Press Ctrl+A, then D to detach
   ```

3. **Using PowerShell (Windows)**
   ```powershell
   Start-Process -FilePath "uv" -ArgumentList "run", "streamlit", "run", "app.py" -WindowStyle Hidden
   ```

### Process Management

Effective process management ensures the application runs smoothly:

1. **Finding Running Processes**
   ```bash
   ps aux | grep streamlit
   ```

2. **Stopping Processes**
   ```bash
   pkill -f streamlit
   ```


3. **Using Process Managers**
   
   For more robust process management, consider using tools like `supervisor` (Linux) or `pm2` (cross-platform):
   
   Install pm2:
   ```bash
   npm install -g pm2
   ```
   
   Start StockTracker with pm2:
   ```bash
   pm2 start app.py --name stocktracker --interpreter python
   pm2 save
   pm2 startup  # To start on system boot
   ```
## 3. Server Deployment

Deploying StockTracker on a Linux server provides a stable environment for team usage or small-scale production.

### Linux Server Setup

1. **Update System Packages**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install Required System Packages**
   ```bash
   sudo apt install -y python3.12 python3.12-venv git nginx docker.io docker-compose
   ```

3. **Create Dedicated User**
   ```bash
   sudo useradd -r -s /bin/false stocktracker
   sudo mkdir -p /opt/stocktracker
   sudo chown stocktracker:stocktracker /opt/stocktracker
   ```

4. **Install uv Package Manager**
   ```bash
   pip install uv
   ```

### Docker Deployment

StockTracker can be deployed using Docker for consistent environments:

1. **Build Docker Image**
   ```bash
   docker build -t stocktracker .
   ```

2. **Run Container**
   ```bash
   docker run -d -p 8501:8501 --name stocktracker-app stocktracker
   ```

3. **Using Docker Compose**
   
   Create a `docker-compose.yml` file:
   ```yaml
   version: '3.8'
   services:
     stocktracker:
       build: .
       ports:
         - "8501:8501"
       volumes:
         - ./data:/app/data
         - ./models:/app/models
       environment:
         - STOCKTRACKER_DATA_DIR=/app/data
         - STOCKTRACKER_MODELS_DIR=/app/models
   ```

   Deploy with:
   ```bash
   docker-compose up -d
   ```

### Reverse Proxy Configuration (nginx)

Configure nginx as a reverse proxy for better performance and security:

1. **Create nginx Configuration**
   ```bash
   sudo nano /etc/nginx/sites-available/stocktracker
   ```

   Add the following configuration:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:8501;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

2. **Enable Site**
   ```bash
   sudo ln -s /etc/nginx/sites-available/stocktracker /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

### SSL/HTTPS Setup

Secure your deployment with SSL certificates using Let's Encrypt:

1. **Install Certbot**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   ```

2. **Obtain SSL Certificate**
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```

3. **Auto-renewal**
   ```bash
   sudo crontab -e
   # Add the following line:
   0 12 * * * /usr/bin/certbot renew --quiet
   ```

### Service Management (systemd)

Manage StockTracker as a system service for automatic startup and recovery:

1. **Create systemd Service File**
   ```bash
   sudo nano /etc/systemd/system/stocktracker.service
   ```

   Add the following content:
   ```ini
   [Unit]
   Description=StockTracker Application
   After=network.target

   [Service]
   Type=simple
   User=stocktracker
   WorkingDirectory=/opt/stocktracker
   ExecStart=/opt/stocktracker/.venv/bin/uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```

2. **Enable and Start Service**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable stocktracker
   sudo systemctl start stocktracker
   ```

3. **Check Service Status**
   ```bash
   sudo systemctl status stocktracker
   ```

4. **View Logs**
   ```bash
   journalctl -u stocktracker -f
   ```
## 4. Cloud Deployment

Deploying StockTracker on cloud platforms provides scalability, reliability, and managed services.

### AWS Deployment

Deploy StockTracker on Amazon Web Services using EC2 or ECS:

1. **EC2 Deployment**
   
   a. Launch an EC2 instance (Ubuntu 22.04 LTS recommended)
   b. Connect to your instance via SSH
   c. Follow the Linux server setup instructions above
   d. Configure security groups to allow HTTP/HTTPS traffic

2. **ECS Deployment**
   
   a. Create an ECS cluster
   b. Build and push the Docker image to ECR:
      ```bash
      aws ecr create-repository --repository-name stocktracker
      docker build -t stocktracker .
      docker tag stocktracker:latest <your-account-id>.dkr.ecr.<region>.amazonaws.com/stocktracker:latest
      docker push <your-account-id>.dkr.ecr.<region>.amazonaws.com/stocktracker:latest
      ```
   c. Create an ECS task definition
   d. Create an ECS service

3. **Using Elastic Beanstalk**
   
   a. Install the EB CLI:
      ```bash
      pip install awsebcli
      ```
   b. Initialize your application:
      ```bash
      eb init
      ```
   c. Create and deploy your environment:
      ```bash
      eb create stocktracker-env
      eb deploy
      ```

### Azure Deployment

Deploy StockTracker on Microsoft Azure using Virtual Machines or Container Instances:

1. **Azure VM Deployment**
   
   a. Create an Ubuntu Virtual Machine
   b. Connect via SSH
   c. Follow the Linux server setup instructions above

2. **Azure Container Instances**
   
   a. Build and push image to Azure Container Registry:
      ```bash
      az acr create --resource-group myResourceGroup --name mycontainerregistry --sku Basic
      docker build -t stocktracker .
      az acr login --name mycontainerregistry
      docker tag stocktracker mycontainerregistry.azurecr.io/stocktracker
      docker push mycontainerregistry.azurecr.io/stocktracker
      ```
   b. Deploy container:
      ```bash
      az container create --resource-group myResourceGroup \
        --name stocktracker-container \
        --image mycontainerregistry.azurecr.io/stocktracker \
        --dns-name-label stocktracker-app \
        --ports 8501
      ```

3. **Azure App Service**
   
   a. Create a Web App for Containers
   b. Configure to use your Docker image
   c. Set environment variables as needed

### Google Cloud Deployment

Deploy StockTracker on Google Cloud Platform using Compute Engine or Cloud Run:

1. **Compute Engine Deployment**
   
   a. Create a Compute Engine instance (Ubuntu recommended)
   b. Connect via SSH
   c. Follow the Linux server setup instructions above

2. **Cloud Run Deployment**
   
   a. Build and push image to Google Container Registry:
      ```bash
      docker build -t stocktracker .
      docker tag stocktracker gcr.io/[PROJECT-ID]/stocktracker
      docker push gcr.io/[PROJECT-ID]/stocktracker
      ```
   b. Deploy to Cloud Run:
      ```bash
      gcloud run deploy --image gcr.io/[PROJECT-ID]/stocktracker --platform managed
      ```

3. **Google Kubernetes Engine**
   
   a. Create a GKE cluster
   b. Deploy using Kubernetes manifests
   c. Configure load balancing and autoscaling

### Platform-specific Considerations

1. **Resource Allocation**
   - Monitor CPU and memory usage
   - Scale resources based on demand
   - Use auto-scaling when available

2. **Data Persistence**
   - Use cloud storage services for data and model persistence
   - Configure regular backups
   - Implement disaster recovery plans

3. **Networking**
   - Configure firewalls and security groups
   - Use private networks when possible
   - Implement load balancing for high availability

4. **Monitoring and Logging**
   - Integrate with cloud monitoring services
   - Set up alerts for critical metrics
   - Centralize logs for analysis

5. **Cost Optimization**
   - Use spot instances or preemptible VMs when appropriate
   - Monitor resource usage to avoid over-provisioning
   - Take advantage of reserved instances for predictable workloads
   User=stocktracker
   WorkingDirectory=/opt/stocktracker
   ExecStart=/opt/stocktracker/.venv/bin/uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   Restart=always
   RestartSec=10
## 5. Containerization

Containerization provides a consistent and portable deployment method for StockTracker across different environments.

### Dockerfile Explanation

Create a `Dockerfile` in the project root with the following content:

```dockerfile
# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install uv package manager
RUN pip install uv

# Install project dependencies
RUN uv sync

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the application
CMD ["uv", "run", "streamlit", "run", "app.py"]
```

Key components of the Dockerfile:
- Uses Python 3.12 slim image for a smaller footprint
- Installs necessary system dependencies for some Python packages
- Copies all project files to the container
- Uses uv for faster dependency installation
- Exposes port 8501 for the Streamlit web interface
- Sets environment variables for Streamlit configuration
- Defines the command to run the application

### Building Docker Images

1. **Build the Image**
   ```bash
   docker build -t stocktracker .
   ```

2. **Build with Custom Tag**
   ```bash
   docker build -t stocktracker:v1.0 .
   ```

3. **Build with No Cache**
   ```bash
   docker build --no-cache -t stocktracker .
   ```

### Docker Compose Setup

Create a `docker-compose.yml` file for multi-container setups and easier management:

```yaml
version: '3.8'

services:
  stocktracker:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - STOCKTRACKER_DATA_DIR=/app/data
      - STOCKTRACKER_MODELS_DIR=/app/models
      - STOCKTRACKER_LOG_LEVEL=INFO
    restart: unless-stopped
    networks:
      - stocktracker-network

networks:
  stocktracker-network:
    driver: bridge
```

To use Docker Compose:
1. **Start Services**
   ```bash
   docker-compose up -d
   ```

2. **View Logs**
   ```bash
   docker-compose logs -f
   ```

3. **Stop Services**
   ```bash
   docker-compose down
   ```

### Volume Management

Proper volume management ensures data persistence across container restarts:

1. **Data Volume**
   - Mount local directory for stock data storage
   - Ensures downloaded data persists between container restarts

2. **Models Volume**
   - Mount local directory for trained model storage
   - Preserves trained models across deployments

3. **Logs Volume**
   - Mount local directory for log storage
   - Enables log analysis and debugging

Example volume configuration:
```bash
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --name stocktracker-app \
  stocktracker
```

### Environment Variables in Containers

Configure StockTracker behavior using environment variables:


| Environment Variable | Description | Default Value |
|---------------------|-------------|---------------|
| `STOCKTRACKER_DATA_DIR` | Directory for storing stock data | `/app/data` |
| `STOCKTRACKER_MODELS_DIR` | Directory for saving trained models | `/app/models` |
| `STOCKTRACKER_LOG_LEVEL` | Application logging level | `INFO` |
| `STREAMLIT_SERVER_PORT` | Streamlit server port | `8501` |
| `STREAMLIT_SERVER_ADDRESS` | Streamlit server address | `0.0.0.0` |

Set environment variables in Docker:
```bash
docker run -d \
  -p 8501:8501 \
  -e STOCKTRACKER_DATA_DIR=/app/data \
  -e STOCKTRACKER_LOG_LEVEL=DEBUG \
  --name stocktracker-app \
  stocktracker
```

Or in Docker Compose:
```yaml
environment:
  - STOCKTRACKER_DATA_DIR=/app/data
  - STOCKTRACKER_LOG_LEVEL=DEBUG
```
## 6. Performance Optimization

Optimizing StockTracker's performance ensures faster response times and better resource utilization.

### Data Fetching Optimization

Since StockTracker uses akshare to fetch stock data, optimizing data fetching is crucial:

1. **Connection Pooling**
   - Reuse connections to reduce overhead
   - Configure appropriate timeout values

2. **Batch Processing**
   - Fetch data for multiple stocks in parallel
   - Use threading or async operations where possible

3. **Rate Limiting Compliance**
   - Respect API rate limits to avoid throttling
   - Implement exponential backoff for failed requests

Example optimization in data fetching:
```python
import concurrent.futures
import data.fetcher as data_fetcher

def fetch_multiple_stocks(symbols):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {
            executor.submit(data_fetcher.get_stock_data, symbol): symbol 
            for symbol in symbols
        }
        results = {}
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                print(f'{symbol} generated an exception: {exc}')
        return results
```

### Caching Strategies

Implement caching to reduce redundant data fetching and computation:

1. **In-Memory Caching**
   - Use Python's `functools.lru_cache` for function results
   - Cache frequently accessed stock data

2. **File-Based Caching**
   - Store downloaded data in local files
   - Check file modification time to determine cache validity

3. **Redis Caching** (for distributed deployments)
   - Use Redis for shared caching across multiple instances
   - Cache model predictions and technical indicators

Example LRU cache implementation:
```python
from functools import lru_cache
import pandas as pd

@lru_cache(maxsize=128)
def get_cached_stock_data(symbol, period="daily", start_date=None):
    # This would call the actual data fetching function
    return data_fetcher.get_stock_data(symbol, period, start_date)
```

Example Redis caching:
```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_or_fetch(symbol):
    cached_data = r.get(f"stock_data:{symbol}")
    if cached_data:
        return pd.read_json(cached_data)
    
    # Fetch fresh data
    data = data_fetcher.get_stock_data(symbol)
    
    # Cache for 1 hour
    r.setex(f"stock_data:{symbol}", 3600, data.to_json())
    return data
```

### Model Performance Optimization

Optimize machine learning model performance:

1. **GPU Acceleration**
   - Ensure TensorFlow can access GPU resources
   - Monitor GPU memory usage

2. **Model Quantization**
   - Reduce model size for faster loading
   - Use TensorFlow Lite for deployment

3. **Prediction Batching**
   - Batch predictions for multiple stocks
   - Reduce inference overhead

### Resource Allocation

Proper resource allocation ensures optimal performance:

1. **CPU Allocation**
   - Reserve CPU cores for data fetching and model training
   - Use CPU affinity for critical processes

2. **Memory Management**
   - Monitor memory usage to prevent out-of-memory errors
   - Implement data streaming for large datasets

3. **Disk I/O Optimization**
   - Use SSD storage for data and model files
   - Implement efficient data serialization (e.g., Parquet)

### Load Balancing

For high-traffic deployments:

1. **Horizontal Scaling**
   - Deploy multiple instances behind a load balancer
   - Use sticky sessions for user session consistency

2. **Reverse Proxy Load Balancing**
   - Configure nginx for load distribution
   - Implement health checks for instance monitoring

Example nginx load balancing configuration:
```nginx
upstream stocktracker_backend {
    least_conn;
    server 192.168.1.10:8501;
    server 192.168.1.11:8501;
    server 192.168.1.12:8501;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://stocktracker_backend;
## 7. Monitoring and Maintenance

Proper monitoring and maintenance ensure StockTracker runs smoothly and reliably.

### Log Management

Effective log management helps with debugging and monitoring:

1. **Application Logging**
   
   StockTracker uses Python's built-in logging module. Configure logging levels:
   ```python
   import logging
   
   # Set logging level
   logging.basicConfig(level=logging.INFO)
   
   # Or via environment variable
   import os
   log_level = os.getenv('STOCKTRACKER_LOG_LEVEL', 'INFO')
   logging.basicConfig(level=getattr(logging, log_level))
   ```

2. **Log Rotation**
   
   Implement log rotation to prevent disk space issues:
   ```bash
   # Using logrotate (Linux)
   sudo nano /etc/logrotate.d/stocktracker
   
   # Add the following content:
   /var/log/stocktracker/*.log {
       daily
       rotate 30
       compress
       delaycompress
       missingok
       notifempty
       create 644 stocktracker stocktracker
   }
   ```

3. **Centralized Logging**
   
   For distributed deployments, use centralized logging:
   - ELK Stack (Elasticsearch, Logstash, Kibana)
   - Fluentd with Elasticsearch
   - Cloud logging services (CloudWatch, Azure Monitor, Stackdriver)

### Health Checks

Implement health checks to monitor application status:

1. **Basic Health Check**
   
   Add a health check endpoint to your application:
   ```python
   import streamlit as st
   
   # Add to app.py
   def health_check():
       return {"status": "healthy", "timestamp": str(pd.Timestamp.now())}
   
   # In a separate endpoint or function
   if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "health":
       print(health_check())
   ```

2. **Docker Health Check**
   
   Add health check to Dockerfile:
   ```dockerfile
   HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
     CMD curl -f http://localhost:8501/health || exit 1
   ```

3. **Kubernetes Liveness and Readiness Probes**
   
   Example Kubernetes probe configuration:
   ```yaml
   livenessProbe:
     httpGet:
       path: /health
       port: 8501
     initialDelaySeconds: 30
     periodSeconds: 10
   
   readinessProbe:
     httpGet:
       path: /health
       port: 8501
     initialDelaySeconds: 5
     periodSeconds: 5
   ```

### Backup and Recovery

Implement backup and recovery procedures for data and models:

1. **Data Backup**
   
   Regularly backup stock data:
   ```bash
   # Create backup script
   #!/bin/bash
   BACKUP_DIR="/backup/stocktracker/$(date +%Y%m%d)"
   mkdir -p $BACKUP_DIR
   cp -r /opt/stocktracker/data $BACKUP_DIR/
   cp -r /opt/stocktracker/models $BACKUP_DIR/
   
   # Schedule with cron
   0 2 * * * /opt/stocktracker/scripts/backup.sh
   ```

2. **Model Backup**
   
   Backup trained models:
   ```bash
   # Include model files in backup
   tar -czf models_backup_$(date +%Y%m%d).tar.gz /opt/stocktracker/models/
   ```

3. **Cloud Backup**
   
   Use cloud storage for offsite backups:
   ```bash
   # AWS S3 backup
   aws s3 sync /opt/stocktracker/data s3://stocktracker-backup/data/
   aws s3 sync /opt/stocktracker/models s3://stocktracker-backup/models/
   
   # Google Cloud Storage
   gsutil -m rsync -r /opt/stocktracker/data gs://stocktracker-backup/data/
   ```

4. **Recovery Procedures**
   
   Document recovery steps:
   ```bash
   # Restore from backup
   tar -xzf models_backup_20230101.tar.gz -C /opt/stocktracker/
   ```

### Update Procedures

Follow proper procedures when updating StockTracker:

1. **Version Control**
   
   Use Git for version management:
   ```bash
   # Pull latest changes
   git pull origin main
   
   # Check for changes
   git diff HEAD@{1} HEAD
## 8. Security Considerations

Security is critical for protecting data and ensuring the integrity of the StockTracker application.

### Access Control

Implement proper access control measures:

1. **User Authentication**
   
   For production deployments, add authentication to the Streamlit application:
   ```python
   import streamlit as st
   import streamlit_authenticator as stauth
   
   # Configuration for authenticator
   config = {
       'credentials': {
           'usernames': {
               'admin': {
                   'name': 'Administrator',
                   'password': 'hashed_password_here'
               }
           }
       },
       'cookie': {
           'name': 'stocktracker_cookie',
           'key': 'stocktracker_cookie_key',
           'expiry_days': 30
       }
   }
   
   authenticator = stauth.Authenticate(
       config['credentials'],
       config['cookie']['name'],
       config['cookie']['key'],
       config['cookie']['expiry_days']
   )
   
   name, authentication_status, username = authenticator.login('Login', 'main')
   
   if authentication_status:
       # User is authenticated, show application
       st.success(f'Welcome {name}')
   elif authentication_status is False:
       st.error('Username/password is incorrect')
   elif authentication_status is None:
       st.warning('Please enter your username and password')
   ```

2. **Role-Based Access Control**
   
   Implement different access levels:
   - Admin: Full access to all features
   - User: Limited access to prediction and analysis features
   - Guest: Read-only access

3. **API Key Management**
   
   If integrating with external services, securely manage API keys:
   ```python
   import os
   from dotenv import load_dotenv
   
   # Load environment variables from .env file
   load_dotenv()
   
   # Access API keys
   api_key = os.getenv('API_KEY')
   ```

### Data Protection

Protect sensitive data and ensure privacy:

1. **Data Encryption**
   
   Encrypt data at rest and in transit:
   - Use HTTPS/TLS for data in transit
   - Consider encrypting sensitive data files at rest

2. **Personal Data Handling**
   
   If storing user data:
   - Implement data minimization principles
   - Provide data export and deletion capabilities
   - Comply with relevant privacy regulations (GDPR, CCPA)

3. **Secure File Permissions**
   
   Set appropriate file permissions:
   ```bash
   # Set ownership
   sudo chown -R stocktracker:stocktracker /opt/stocktracker/
   
   # Set permissions
   sudo chmod -R 750 /opt/stocktracker/
   sudo chmod 600 /opt/stocktracker/.env
   ```

### Network Security

Implement network-level security measures:

1. **Firewall Configuration**
   
   Configure firewall rules to restrict access:
   ```bash
   # Using ufw (Ubuntu)
   sudo ufw allow ssh
   sudo ufw allow http
   sudo ufw allow https
   sudo ufw enable
   ```

2. **Reverse Proxy Security**
   
   Configure nginx security headers:
   ```nginx
   server {
       listen 443 ssl;
       server_name your-domain.com;
       
       # Security headers
       add_header X-Frame-Options "SAMEORIGIN" always;
       add_header X-XSS-Protection "1; mode=block" always;
       add_header X-Content-Type-Options "nosniff" always;
       add_header Referrer-Policy "no-referrer-when-downgrade" always;
       add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
       
       # SSL configuration
       ssl_certificate /path/to/certificate.crt;
       ssl_certificate_key /path/to/private.key;
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
       ssl_prefer_server_ciphers off;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

3. **SSH Security**
   
   Secure SSH access to servers:
   ```bash
   # Edit SSH configuration
   sudo nano /etc/ssh/sshd_config
   
   # Recommended settings:
   Port 2222
   PermitRootLogin no
   PasswordAuthentication no
   PubkeyAuthentication yes
   AllowUsers specific_user
   ```

### Regular Security Updates

Maintain system and application security:

1. **System Updates**
   
   Regularly update the operating system:
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt upgrade -y
   
   # Schedule automatic updates
   sudo apt install unattended-upgrades
   sudo dpkg-reconfigure -plow unattended-upgrades
   ```

2. **Dependency Updates**
   
   Keep Python dependencies up to date:
   ```bash
   # Check for outdated packages
   uv pip list --outdated
   
   # Update all packages
   uv sync --upgrade
   ```

3. **Security Scanning**
   
   Regularly scan for vulnerabilities:
   ```bash
   # Install and run bandit for Python security issues
   pip install bandit
   bandit -r .
   
   # Docker image scanning
   docker scan stocktracker
   ```

4. **Penetration Testing**
   
   Periodically perform security assessments:
   - Use tools like OWASP ZAP for web application testing
   - Conduct internal security reviews
   - Engage third-party security auditors for critical deployments

### Incident Response

Prepare for security incidents:

1. **Monitoring and Alerting**
   
   Set up security monitoring:
   - Monitor authentication logs
   - Set up intrusion detection systems
   - Configure alerts for suspicious activities

2. **Incident Response Plan**
   
   Document procedures for security incidents:
   - Identification and containment
   - Eradication and recovery
   - Post-incident analysis and reporting

3. **Backup and Recovery**
   
   Ensure secure backups are available:
   - Regular backup testing
   - Secure backup storage
   - Encryption of backup data

### Performance Monitoring

Monitor key performance metrics:

1. **Response Times**
   - Track API response times
   - Set alerts for performance degradation

2. **Resource Utilization**
   - Monitor CPU, memory, and disk usage
   - Implement auto-scaling based on metrics

3. **Error Rates**
   - Track failed requests and exceptions
   - Implement error rate alerts