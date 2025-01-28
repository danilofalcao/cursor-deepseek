# DeepSeek API Proxy

A high-performance HTTP/2-enabled proxy server designed specifically to enable Cursor IDE's Composer to use DeepSeek's and OpenRouter's language models. This proxy translates OpenAI-compatible API requests to DeepSeek/OpenRouter API format, allowing Cursor's Composer and other OpenAI API-compatible tools to seamlessly work with these models.

## Primary Use Case

This proxy was created to enable Cursor IDE users to leverage DeepSeek's and OpenRouter's powerful language models through Cursor's Composer interface as an alternative to OpenAI's models. By running this proxy locally, you can configure Cursor's Composer to use these models for AI assistance, code generation, and other AI features. It handles all the necessary request/response translations and format conversions to make the integration seamless.

## Features

- HTTP/2 support for improved performance
- Full CORS support
- Streaming responses
- Support for function calling/tools
- Automatic message format conversion
- Compatible with OpenAI API client libraries
- API key validation for secure access
- Support for both DeepSeek and OpenRouter endpoints
- Docker container support

## Prerequisites

- Cursor Pro Subscription
- Go 1.19 or higher
- DeepSeek API key and/or OpenRouter API key
- Public Endpoint

## Installation

1. Clone the repository
2. Install dependencies:
```bash
go mod download
```

### Docker Installation

1. Build the Docker image:
```bash
docker build -t cursor-deepseek .
```

2. Configure environment variables:
   - Copy the example configuration:
   ```bash
   cp .env.example .env
   ```
   - Edit `.env` and add your API key(s)

3. Run the container:
```bash
docker run -p 9000:9000 --env-file .env cursor-deepseek
```

## Configuration

The repository includes an `.env.example` file showing the required environment variables. To configure:

1. Copy the example configuration:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API key(s):
```bash
# For DeepSeek (required for chat and coder models)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# For OpenRouter (required for openrouter model)
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Note: You can configure either one or both API keys depending on which models you plan to use.

## Usage

Start the proxy server with one of the following commands:

```bash
# For DeepSeek Chat model (default)
go run proxy.go -model chat

# For DeepSeek Coder model
go run proxy.go -model coder

# For OpenRouter DeepSeek model
go run proxy.go -model openrouter
```

The server will start on port 9000 by default.

Use the proxy with your OpenAI API clients by setting the base URL to `http://your-public-endpoint:9000/v1`

### Supported Models

- `gpt-4o` maps to:
  - DeepSeek Chat model (`deepseek-chat`) when using `-model chat`
  - DeepSeek Coder model (`deepseek-coder`) when using `-model coder`
  - DeepSeek OpenRouter model (`deepseek/deepseek-chat`) when using `-model openrouter`

### Supported Endpoints

- `/v1/chat/completions` - Chat completions endpoint
- `/v1/models` - Models listing endpoint

## Dependencies

- `github.com/joho/godotenv` - Environment variable management
- `golang.org/x/net` - HTTP/2 support

## Security

- The proxy includes CORS headers for cross-origin requests
- API keys are required and validated against environment variables
- Secure handling of request/response data
- Strict API key validation for all requests
- HTTPS support through HTTP/2
- Environment variables are never committed to the repository

## License

This project is licensed under the GNU General Public License v2.0 (GPLv2). See the [LICENSE.md](LICENSE.md) file for details.
