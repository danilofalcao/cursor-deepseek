# Build stage
FROM golang:1.21-alpine AS builder

# Add build argument to specify which proxy to build
ARG PROXY_VARIANT=deepseek

# Install necessary build tools
RUN apk add --no-cache git

# Set working directory
WORKDIR /app

# Copy go mod files first for better caching
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source files
COPY . .

# Build the application based on the selected variant
RUN if [ "$PROXY_VARIANT" = "openrouter" ]; then \
        CGO_ENABLED=0 GOOS=linux go build -o proxy proxy-openrouter.go; \
    else \
        CGO_ENABLED=0 GOOS=linux go build -o proxy proxy.go; \
    fi

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS requests
RUN apk --no-cache add ca-certificates

# Set working directory
WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/proxy .

# Note: Environment variables should be provided at runtime
# Example: docker run -p 9000:9000 --env-file .env cursor-deepseek

# Expose port 9000
EXPOSE 9000

# Run the application
CMD ["./proxy"] 
