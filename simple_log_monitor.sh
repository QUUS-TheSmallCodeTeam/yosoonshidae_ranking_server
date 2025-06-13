#!/bin/sh

# Simple log monitor that filters useful logs
# Usage: ./simple_log_monitor.sh

# Get Python server PID
PID=$(ps aux | grep "python.*uvicorn" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "Server not found!"
    exit 1
fi

echo "Monitoring server PID: $PID"
echo "Filtering out GET / requests..."
echo "Press Ctrl+C to stop"

# Monitor stderr for application logs and stdout for HTTP logs
(cat /proc/$PID/fd/2 & cat /proc/$PID/fd/1 | grep -v -E "(GET / HTTP/1.1|GET /favicon)") | while read line; do
    echo "$line"
    echo "$(date '+%Y-%m-%d %H:%M:%S'): $line" >> error.log
    
    # Keep error.log manageable (last 500 lines)
    # Use a simple counter instead of RANDOM (which is bash-specific)
    if [ ! -f .log_counter ]; then echo "0" > .log_counter; fi
    counter=$(cat .log_counter)
    counter=$((counter + 1))
    echo "$counter" > .log_counter
    
    if [ $((counter % 50)) -eq 0 ]; then  # Check every 50 lines
        if [ $(wc -l < error.log) -gt 500 ]; then
            tail -500 error.log > error.log.tmp && mv error.log.tmp error.log
        fi
    fi
done 