#!/bin/bash
# Test script to verify Docker container proxy configuration

echo "Testing Docker container proxy configuration..."
echo "========================================="

# Run a test container with the proxy settings
docker run --rm \
  -e HTTP_PROXY=http://host.docker.internal:7890 \
  -e HTTPS_PROXY=http://host.docker.internal:7890 \
  -e http_proxy=http://host.docker.internal:7890 \
  -e https_proxy=http://host.docker.internal:7890 \
  -e ALL_PROXY=socks5://host.docker.internal:7890 \
  -e all_proxy=socks5://host.docker.internal:7890 \
  -e NO_PROXY=localhost,127.0.0.1,host.docker.internal \
  -e no_proxy=localhost,127.0.0.1,host.docker.internal \
  alpine:latest sh -c '
    echo "Container Environment Variables:"
    echo "HTTP_PROXY=$HTTP_PROXY"
    echo "HTTPS_PROXY=$HTTPS_PROXY"
    echo "ALL_PROXY=$ALL_PROXY"
    echo "NO_PROXY=$NO_PROXY"
    echo ""
    echo "Testing connectivity through proxy:"
    if wget -q -O /dev/null --timeout=5 https://www.google.com 2>/dev/null; then
      echo "✅ Successfully connected through proxy"
    else
      echo "❌ Failed to connect through proxy"
    fi
    echo ""
    echo "Testing direct connection to host.docker.internal:"
    if wget -q -O /dev/null --timeout=5 http://host.docker.internal:2999/ 2>/dev/null; then
      echo "✅ Successfully connected to host.docker.internal:2999"
    else
      echo "❌ Failed to connect to host.docker.internal:2999"
    fi
  '