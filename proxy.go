package main

import (
        "bufio"
        "bytes"
        "compress/gzip"
        "compress/flate"
        "encoding/json"
        "fmt"
        "io"
        "log"
        "net/http"
        "net/http/httputil"
        "os"
        "strings"
        "time"

        "github.com/andybalholm/brotli"
        "github.com/joho/godotenv"
        "golang.org/x/net/http2"
)

const (
        deepseekEndpoint  = "https://api.deepseek.com"
        deepseekChatModel = "deepseek-chat"
        gpt4oModel       = "gpt-4o"
)

var deepseekAPIKey string

func init() {
        // Load .env file
        if err := godotenv.Load(); err != nil {
                log.Printf("Warning: .env file not found or error loading it: %v", err)
        }

        // Get DeepSeek API key
        deepseekAPIKey = os.Getenv("DEEPSEEK_API_KEY")
        if deepseekAPIKey == "" {
                log.Fatal("DEEPSEEK_API_KEY environment variable is required")
        }
}

// Models response structure
type ModelsResponse struct {
        Object string  `json:"object"`
        Data   []Model `json:"data"`
}

type Model struct {
        ID      string `json:"id"`
        Object  string `json:"object"`
        Created int64  `json:"created"`
        OwnedBy string `json:"owned_by"`
}

// OpenAI compatible request structure
type ChatRequest struct {
        Model       string        `json:"model"`
        Messages    []Message     `json:"messages"`
        Stream      bool          `json:"stream"`
        Functions   []Function    `json:"functions,omitempty"`
        Tools      []Tool        `json:"tools,omitempty"`
        ToolChoice interface{}   `json:"tool_choice,omitempty"`
        Temperature *float64     `json:"temperature,omitempty"`
        MaxTokens   *int         `json:"max_tokens,omitempty"`
}

type Message struct {
        Role       string      `json:"role"`
        Content    string      `json:"content"`
        ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
        ToolCallID string     `json:"tool_call_id,omitempty"`
        Name       string     `json:"name,omitempty"`
}

type Function struct {
        Name        string `json:"name"`
        Description string `json:"description"`
        Parameters  any    `json:"parameters"`
}

type Tool struct {
        Type     string   `json:"type"`
        Function Function `json:"function"`
}

type ToolCall struct {
        ID       string `json:"id"`
        Type     string `json:"type"`
        Function struct {
                Name      string `json:"name"`
                Arguments string `json:"arguments"`
        } `json:"function"`
}

func convertToolChoice(choice interface{}) string {
        if choice == nil {
                return ""
        }

        // If string "auto" or "none"
        if str, ok := choice.(string); ok {
                switch str {
                case "auto", "none":
                        return str
                }
        }

        // Try to parse as map for function call
        if choiceMap, ok := choice.(map[string]interface{}); ok {
                if choiceMap["type"] == "function" {
                        return "auto" // DeepSeek doesn't support specific function selection, default to auto
                }
        }

        return ""
}

func convertMessages(messages []Message) []Message {
        converted := make([]Message, len(messages))
        for i, msg := range messages {
                log.Printf("Converting message %d - Role: %s", i, msg.Role)
                converted[i] = msg

                // Handle assistant messages with tool calls
                if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
                        log.Printf("Processing assistant message with %d tool calls", len(msg.ToolCalls))
                        // DeepSeek expects tool_calls in a specific format
                        toolCalls := make([]ToolCall, len(msg.ToolCalls))
                        for j, tc := range msg.ToolCalls {
                                toolCalls[j] = ToolCall{
                                        ID:       tc.ID,
                                        Type:     "function",
                                        Function: tc.Function,
                                }
                                log.Printf("Tool call %d - ID: %s, Function: %s", j, tc.ID, tc.Function.Name)
                        }
                        converted[i].ToolCalls = toolCalls
                }

                // Handle function response messages
                if msg.Role == "function" {
                        log.Printf("Converting function response to tool response")
                        // Convert to tool response format
                        converted[i].Role = "tool"
                }
        }

        // Log the final converted messages
        for i, msg := range converted {
                log.Printf("Final message %d - Role: %s, Content: %s", i, msg.Role, truncateString(msg.Content, 50))
                if len(msg.ToolCalls) > 0 {
                        log.Printf("Message %d has %d tool calls", i, len(msg.ToolCalls))
                }
        }

        return converted
}

func truncateString(s string, maxLen int) string {
        if len(s) <= maxLen {
                return s
        }
        return s[:maxLen] + "..."
}

// DeepSeek request structure
type DeepSeekRequest struct {
        Model       string        `json:"model"`
        Messages    []Message     `json:"messages"`
        Stream      bool          `json:"stream"`
        Temperature float64       `json:"temperature,omitempty"`
        MaxTokens   int          `json:"max_tokens,omitempty"`
        Tools       []Tool       `json:"tools,omitempty"`
        ToolChoice  string       `json:"tool_choice,omitempty"`
}

func main() {
        log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds | log.Lshortfile)

        server := &http.Server{
                Addr:    ":9000",
                Handler: http.HandlerFunc(proxyHandler),
        }

        // Enable HTTP/2 support
        http2.ConfigureServer(server, &http2.Server{})

        log.Printf("Starting proxy server on %s", server.Addr)
        if err := server.ListenAndServe(); err != nil {
                log.Fatalf("Server failed: %v", err)
        }
}

func enableCors(w http.ResponseWriter, r *http.Request) {
        // Allow requests from any origin
        w.Header().Set("Access-Control-Allow-Origin", "*")
        // Allow specific headers
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
        // Allow specific methods
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        // Allow credentials
        w.Header().Set("Access-Control-Allow-Credentials", "true")
        // Set max age for preflight requests
        w.Header().Set("Access-Control-Max-Age", "3600")
}

func proxyHandler(w http.ResponseWriter, r *http.Request) {
        // Log incoming request
        log.Printf("Incoming request: %s %s", r.Method, r.URL.Path)
        reqDump, err := httputil.DumpRequest(r, true)
        if err == nil {
                log.Printf("Request details:\n%s", string(reqDump))
        }

        // Enable CORS for all requests
        enableCors(w, r)

        // Handle OPTIONS requests
        if r.Method == "OPTIONS" {
                w.WriteHeader(http.StatusOK)
                return
        }

        // Handle models endpoint
        if r.URL.Path == "/v1/models" {
                handleModelsRequest(w)
                return
        }

        // Only handle API requests with /v1/ prefix
        if !strings.HasPrefix(r.URL.Path, "/v1/") {
                log.Printf("Invalid path: %s", r.URL.Path)
                http.Error(w, "Not found", http.StatusNotFound)
                return
        }

        // Read the original request body
        body, err := io.ReadAll(r.Body)
        if err != nil {
                log.Printf("Error reading request body: %v", err)
                http.Error(w, "Error reading request", http.StatusInternalServerError)
                return
        }
        // Restore the body for further reading
        r.Body = io.NopCloser(bytes.NewBuffer(body))

        log.Printf("Request body: %s", string(body))

        // Parse the request to check for streaming
        var chatReq ChatRequest
        if err := json.Unmarshal(body, &chatReq); err != nil {
                log.Printf("Error parsing request JSON: %v", err)
                http.Error(w, "Error parsing request", http.StatusBadRequest)
                return
        }

        log.Printf("Requested model: %s", chatReq.Model)

        // Replace gpt-4o model with deepseek-chat
        if chatReq.Model == gpt4oModel {
                chatReq.Model = deepseekChatModel
                log.Printf("Model converted to: %s", deepseekChatModel)
        } else {
                log.Printf("Unsupported model requested: %s", chatReq.Model)
                http.Error(w, fmt.Sprintf("Model %s not supported. Use %s instead.", chatReq.Model, gpt4oModel), http.StatusBadRequest)
                return
        }

        // Convert to DeepSeek request format
        deepseekReq := DeepSeekRequest{
                Model:    deepseekChatModel,
                Messages: convertMessages(chatReq.Messages),
                Stream:   chatReq.Stream,
        }

        // Copy optional parameters if present
        if chatReq.Temperature != nil {
                deepseekReq.Temperature = *chatReq.Temperature
        }
        if chatReq.MaxTokens != nil {
                deepseekReq.MaxTokens = *chatReq.MaxTokens
        }

        // Handle tools/functions
        if len(chatReq.Tools) > 0 {
                deepseekReq.Tools = chatReq.Tools
                if tc := convertToolChoice(chatReq.ToolChoice); tc != "" {
                        deepseekReq.ToolChoice = tc
                }
        } else if len(chatReq.Functions) > 0 {
                // Convert functions to tools format
                tools := make([]Tool, len(chatReq.Functions))
                for i, fn := range chatReq.Functions {
                        tools[i] = Tool{
                                Type: "function",
                                Function: fn,
                        }
                }
                deepseekReq.Tools = tools

                // Convert tool_choice if present
                if tc := convertToolChoice(chatReq.ToolChoice); tc != "" {
                        deepseekReq.ToolChoice = tc
                }
        }

        // Create new request body
        modifiedBody, err := json.Marshal(deepseekReq)
        if err != nil {
                log.Printf("Error creating modified request body: %v", err)
                http.Error(w, "Error creating modified request", http.StatusInternalServerError)
                return
        }

        log.Printf("Modified request body: %s", string(modifiedBody))

        // Create the proxy request to DeepSeek
        targetURL := deepseekEndpoint + r.URL.Path
        if r.URL.RawQuery != "" {
                targetURL += "?" + r.URL.RawQuery
        }

        log.Printf("Forwarding to: %s", targetURL)
        proxyReq, err := http.NewRequest(r.Method, targetURL, bytes.NewReader(modifiedBody))
        if err != nil {
                log.Printf("Error creating proxy request: %v", err)
                http.Error(w, "Error creating proxy request", http.StatusInternalServerError)
                return
        }

        // Copy headers
        copyHeaders(proxyReq.Header, r.Header)

        // Set DeepSeek API key and content type
        proxyReq.Header.Set("Authorization", "Bearer "+deepseekAPIKey)
        proxyReq.Header.Set("Content-Type", "application/json")
        if chatReq.Stream {
                proxyReq.Header.Set("Accept", "text/event-stream")
        }

        log.Printf("Proxy request headers: %v", proxyReq.Header)

        // Create a custom client with keepalive
        client := &http.Client{
                Transport: &http2.Transport{
                        AllowHTTP: true,
                        DialTLS: nil,
                },
                Timeout: 5 * time.Minute,
        }

        // Send the request
        resp, err := client.Do(proxyReq)
        if err != nil {
                log.Printf("Error forwarding request: %v", err)
                http.Error(w, "Error forwarding request", http.StatusBadGateway)
                return
        }
        defer resp.Body.Close()

        log.Printf("DeepSeek response status: %d", resp.StatusCode)
        log.Printf("DeepSeek response headers: %v", resp.Header)

        // Handle error responses
        if resp.StatusCode >= 400 {
                respBody, err := io.ReadAll(resp.Body)
                if err != nil {
                        log.Printf("Error reading error response: %v", err)
                        http.Error(w, "Error reading response", http.StatusInternalServerError)
                        return
                }
                log.Printf("DeepSeek error response: %s", string(respBody))

                // Forward the error response
                for k, v := range resp.Header {
                        w.Header()[k] = v
                }
                w.Header().Set("Content-Type", "application/json")
                w.WriteHeader(resp.StatusCode)
                w.Write(respBody)
                return
        }

        // Handle streaming response
        if chatReq.Stream {
                handleStreamingResponse(w, resp)
                return
        }

        // Handle regular response
        handleRegularResponse(w, resp)
}

func handleStreamingResponse(w http.ResponseWriter, resp *http.Response) {
        // Set headers for SSE
        w.Header().Set("Content-Type", "text/event-stream")
        w.Header().Set("Cache-Control", "no-cache")
        w.Header().Set("Connection", "keep-alive")
        w.WriteHeader(resp.StatusCode)

        // Create a buffered reader for the response body
        reader := bufio.NewReader(resp.Body)
        flusher, ok := w.(http.Flusher)
        if !ok {
                http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
                return
        }

        // Stream each line
        for {
                line, err := reader.ReadBytes('\n')
                if err != nil {
                        if err != io.EOF {
                                log.Printf("Error reading stream: %v", err)
                        }
                        return
                }

                // Skip empty lines
                if len(bytes.TrimSpace(line)) == 0 {
                        continue
                }

                // Log the line for debugging
                lineStr := string(line)
                log.Printf("Stream line: %s", lineStr)

                // Handle data: prefix
                if strings.HasPrefix(lineStr, "data: ") {
                        data := strings.TrimPrefix(lineStr, "data: ")
                        data = strings.TrimSpace(data)

                        // Handle [DONE] message
                        if data == "[DONE]" {
                                if _, err := w.Write([]byte("data: [DONE]\n\n")); err != nil {
                                        log.Printf("Error writing done message: %v", err)
                                        return
                                }
                                flusher.Flush()
                                continue // Keep reading for more messages
                        }

                        // Write the line immediately to maintain streaming
                        if _, err := w.Write([]byte(lineStr + "\n")); err != nil {
                                log.Printf("Error writing stream: %v", err)
                                return
                        }
                        flusher.Flush()

                        // Try to parse as JSON for logging
                        var resp struct {
                                Choices []struct {
                                        Delta struct {
                                                Content    string     `json:"content"`
                                                ToolCalls []ToolCall `json:"tool_calls"`
                                        } `json:"delta"`
                                        FinishReason string `json:"finish_reason"`
                                } `json:"choices"`
                        }
                        if err := json.Unmarshal([]byte(data), &resp); err == nil {
                                if len(resp.Choices) > 0 {
                                        choice := resp.Choices[0]

                                        // Log tool calls but don't stop streaming
                                        if len(choice.Delta.ToolCalls) > 0 {
                                                log.Printf("Found tool call in stream: %+v", choice.Delta.ToolCalls)
                                        }

                                        // Log tool call completion but don't stop streaming
                                        if choice.FinishReason == "tool_calls" {
                                                log.Printf("Tool call finished, continuing stream...")
                                                // Write an additional newline to ensure proper SSE formatting
                                                if _, err := w.Write([]byte("\n")); err != nil {
                                                        log.Printf("Error writing newline: %v", err)
                                                        return
                                                }
                                                flusher.Flush()
                                                continue
                                        }

                                        // Log content
                                        if choice.Delta.Content != "" {
                                                log.Printf("Content: %s", choice.Delta.Content)
                                        }
                                }
                        }
                        continue
                }

                // Write any other line with proper SSE format
                if _, err := w.Write([]byte(lineStr + "\n\n")); err != nil {
                        log.Printf("Error writing stream: %v", err)
                        return
                }
                flusher.Flush()
        }
}

func handleRegularResponse(w http.ResponseWriter, resp *http.Response) {
        // Read and decompress response body
        respBody, err := readResponse(resp)
        if err != nil {
                log.Printf("Error reading response body: %v", err)
                http.Error(w, "Error reading response", http.StatusInternalServerError)
                return
        }
        log.Printf("DeepSeek response body: %s", string(respBody))

        // Copy response headers, skipping compression-related ones
        copyHeaders(w.Header(), resp.Header)

        // Set content type and CORS headers
        w.Header().Set("Content-Type", "application/json; charset=utf-8")
        w.Header().Set("Content-Length", fmt.Sprintf("%d", len(respBody)))

        w.WriteHeader(resp.StatusCode)

        // Write the decompressed response back
        if _, err := w.Write(respBody); err != nil {
                log.Printf("Error writing response: %v", err)
        }
}

func copyHeaders(dst, src http.Header) {
        // Headers to skip
        skipHeaders := map[string]bool{
                "Content-Length":    true,
                "Content-Encoding": true,
                "Transfer-Encoding": true,
                "Connection":       true,
        }

        for k, vv := range src {
                if !skipHeaders[k] {
                        for _, v := range vv {
                                dst.Add(k, v)
                        }
                }
        }
}

func handleModelsRequest(w http.ResponseWriter) {
        currentTime := time.Now().Unix()
        models := ModelsResponse{
                Object: "list",
                Data: []Model{
                        {
                                ID:      gpt4oModel,
                                Object:  "model",
                                Created: currentTime,
                                OwnedBy: "organization-owner",
                        },
                },
        }

        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(models)
}

func readResponse(resp *http.Response) ([]byte, error) {
        var reader io.Reader = resp.Body

        switch resp.Header.Get("Content-Encoding") {
        case "gzip":
                gzReader, err := gzip.NewReader(resp.Body)
                if err != nil {
                        return nil, fmt.Errorf("error creating gzip reader: %v", err)
                }
                defer gzReader.Close()
                reader = gzReader
        case "br":
                reader = brotli.NewReader(resp.Body)
        case "deflate":
                reader = flate.NewReader(resp.Body)
        }

        return io.ReadAll(reader)
}
