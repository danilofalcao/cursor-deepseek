package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/joho/godotenv"
	"golang.org/x/net/http2"
)

const (
	deepseekEndpoint     = "https://api.deepseek.com"
	deepseekBetaEndpoint = "https://api.deepseek.com/beta"
	deepseekChatModel    = "deepseek-chat"
	deepseekCoderModel   = "deepseek-coder"
	gpt4oModel           = "gpt-4o"
)

var deepseekAPIKey string

// Configuration structure
type Config struct {
	endpoint string
	model    string
}

var activeConfig Config

// Global HTTP client with optimized settings
var httpClient = &http.Client{
	Transport: &http2.Transport{
		AllowHTTP: true,
		DialTLS:   nil,
		// Optimize connection pooling
		ReadIdleTimeout:  30 * time.Second,
		PingTimeout:      10 * time.Second,
		WriteByteTimeout: 15 * time.Second,
	},
	Timeout: 5 * time.Minute,
}

var (
	// Buffer pools for various sizes
	smallBufferPool = sync.Pool{
		New: func() interface{} {
			return new(bytes.Buffer)
		},
	}

	largeBufferPool = sync.Pool{
		New: func() interface{} {
			return new(bytes.Buffer)
		},
	}

	// Debug mode flag
	debugMode = os.Getenv("DEBUG") == "true"
)

func getBuffer(size int) *bytes.Buffer {
	var buf *bytes.Buffer
	if size < 1024 {
		buf = smallBufferPool.Get().(*bytes.Buffer)
	} else {
		buf = largeBufferPool.Get().(*bytes.Buffer)
	}
	buf.Reset()
	return buf
}

func putBuffer(buf *bytes.Buffer) {
	if buf.Cap() < 1024 {
		smallBufferPool.Put(buf)
	} else {
		largeBufferPool.Put(buf)
	}
}

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

	// Parse command line arguments
	modelFlag := "chat" // default value
	for i, arg := range os.Args {
		if arg == "-model" && i+1 < len(os.Args) {
			modelFlag = os.Args[i+1]
		}
	}

	// Configure the active endpoint and model based on the flag
	switch modelFlag {
	case "coder":
		activeConfig = Config{
			endpoint: deepseekBetaEndpoint,
			model:    deepseekCoderModel,
		}
	case "chat":
		activeConfig = Config{
			endpoint: deepseekEndpoint,
			model:    deepseekChatModel,
		}
	default:
		log.Printf("Invalid model specified: %s. Using default chat model.", modelFlag)
		activeConfig = Config{
			endpoint: deepseekEndpoint,
			model:    deepseekChatModel,
		}
	}

	log.Printf("Initialized with model: %s using endpoint: %s", activeConfig.model, activeConfig.endpoint)
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
	Model       string      `json:"model"`
	Messages    []Message   `json:"messages"`
	Stream      bool        `json:"stream"`
	Functions   []Function  `json:"functions,omitempty"`
	Tools       []Tool      `json:"tools,omitempty"`
	ToolChoice  interface{} `json:"tool_choice,omitempty"`
	Temperature *float64    `json:"temperature,omitempty"`
	MaxTokens   *int        `json:"max_tokens,omitempty"`
}

type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
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
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream"`
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Tools       []Tool    `json:"tools,omitempty"`
	ToolChoice  string    `json:"tool_choice,omitempty"`
}

func debugLog(format string, args ...interface{}) {
	if debugMode {
		log.Printf(format, args...)
	}
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

func enableCors(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization")
	w.Header().Set("Access-Control-Expose-Headers", "Content-Length")
	w.Header().Set("Access-Control-Allow-Credentials", "true")
}

func proxyHandler(w http.ResponseWriter, r *http.Request) {
	debugLog("Received request: %s %s", r.Method, r.URL.Path)

	if r.Method == "OPTIONS" {
		enableCors(w)
		return
	}

	enableCors(w)

	// Validate API key
	authHeader := r.Header.Get("Authorization")
	if !strings.HasPrefix(authHeader, "Bearer ") {
		debugLog("Missing or invalid Authorization header")
		http.Error(w, "Missing or invalid Authorization header", http.StatusUnauthorized)
		return
	}

	userAPIKey := strings.TrimPrefix(authHeader, "Bearer ")
	if userAPIKey != deepseekAPIKey {
		log.Printf("Invalid API key provided")
		http.Error(w, "Invalid API key", http.StatusUnauthorized)
		return
	}

	// Handle /v1/models endpoint
	if r.URL.Path == "/v1/models" && r.Method == "GET" {
		log.Printf("Handling /v1/models request")
		handleModelsRequest(w)
		return
	}

	// Log headers for debugging
	debugLog("Request headers: %+v", r.Header)

	// Read and log request body for debugging
	var chatReq ChatRequest
	body, err := io.ReadAll(r.Body)
	if err != nil {
		debugLog("Error reading request body: %v", err)
		http.Error(w, "Error reading request", http.StatusBadRequest)
		return
	}
	r.Body = io.NopCloser(bytes.NewBuffer(body))

	if err := json.Unmarshal(body, &chatReq); err != nil {
		log.Printf("Error parsing request JSON: %v", err)
		log.Printf("Raw request body: %s", string(body))
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	log.Printf("Parsed request: %+v", chatReq)

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

	// Restore the body for further reading
	r.Body = io.NopCloser(bytes.NewBuffer(body))

	log.Printf("Request body: %s", string(body))

	// Parse the request to check for streaming - reuse existing chatReq
	if err := json.Unmarshal(body, &chatReq); err != nil {
		log.Printf("Error parsing request JSON: %v", err)
		http.Error(w, "Error parsing request", http.StatusBadRequest)
		return
	}

	log.Printf("Requested model: %s", chatReq.Model)

	// Replace gpt-4o model with the appropriate deepseek model
	if chatReq.Model == gpt4oModel {
		log.Printf("Converting gpt-4o to configured model: %s (endpoint: %s)", activeConfig.model, activeConfig.endpoint)
		chatReq.Model = activeConfig.model
		log.Printf("Model converted to: %s", activeConfig.model)
	} else {
		log.Printf("Unsupported model requested: %s", chatReq.Model)
		http.Error(w, fmt.Sprintf("Model %s not supported. Use %s instead.", chatReq.Model, gpt4oModel), http.StatusBadRequest)
		return
	}

	// Convert to DeepSeek request format
	deepseekReq := DeepSeekRequest{
		Model:    activeConfig.model, // Ensure we use the configured model
		Messages: convertMessages(chatReq.Messages),
		Stream:   chatReq.Stream,
	}

	log.Printf("Creating DeepSeek request with model: %s at endpoint: %s", deepseekReq.Model, activeConfig.endpoint)

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
				Type:     "function",
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
	targetURL := activeConfig.endpoint + r.URL.Path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}

	log.Printf("Using endpoint %s with model %s", activeConfig.endpoint, activeConfig.model)
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

	// Add Accept-Language header from request
	if acceptLanguage := r.Header.Get("Accept-Language"); acceptLanguage != "" {
		proxyReq.Header.Set("Accept-Language", acceptLanguage)
	}

	log.Printf("Proxy request headers: %v", proxyReq.Header)

	// Use the global client instead of creating a new one
	resp, err := httpClient.Do(proxyReq)
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
		handleStreamingResponse(w, r, resp)
		return
	}

	// Handle regular response
	handleRegularResponse(w, resp)
}

func handleStreamingResponse(w http.ResponseWriter, r *http.Request, resp *http.Response) {
	debugLog("Starting streaming response handling")
	debugLog("Response status: %d", resp.StatusCode)
	debugLog("Response headers: %+v", resp.Header)

	// Set headers for streaming response
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(resp.StatusCode)

	// Create a buffered reader for the response body
	reader := bufio.NewReader(resp.Body)

	// Create a context with cancel for cleanup
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	// Start a goroutine to send heartbeats
	go func() {
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				// Send a heartbeat comment
				if _, err := w.Write([]byte(": heartbeat\n\n")); err != nil {
					log.Printf("Error sending heartbeat: %v", err)
					cancel()
					return
				}
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			log.Printf("Context cancelled, ending stream")
			return
		default:
			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err == io.EOF {
					continue
				}
				log.Printf("Error reading stream: %v", err)
				cancel()
				return
			}

			// Skip empty lines
			if len(bytes.TrimSpace(line)) == 0 {
				continue
			}

			// Write the line to the response
			if _, err := w.Write(line); err != nil {
				log.Printf("Error writing to response: %v", err)
				cancel()
				return
			}

			// Flush the response writer
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			} else {
				log.Printf("Warning: ResponseWriter does not support Flush")
			}
		}
	}
}

func handleRegularResponse(w http.ResponseWriter, resp *http.Response) {
	debugLog("Handling regular (non-streaming) response")
	debugLog("Response status: %d", resp.StatusCode)
	debugLog("Response headers: %+v", resp.Header)

	// Read and log response body
	body, err := readResponse(resp)
	if err != nil {
		debugLog("Error reading response: %v", err)
		http.Error(w, "Error reading response from upstream", http.StatusInternalServerError)
		return
	}

	debugLog("Original response body: %s", string(body))

	// Parse the DeepSeek response
	var deepseekResp struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			Index        int     `json:"index"`
			Message      Message `json:"message"`
			FinishReason string  `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &deepseekResp); err != nil {
		debugLog("Error parsing DeepSeek response: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	// Convert to OpenAI format
	openAIResp := struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			Index        int     `json:"index"`
			Message      Message `json:"message"`
			FinishReason string  `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}{
		ID:      deepseekResp.ID,
		Object:  "chat.completion",
		Created: deepseekResp.Created,
		Model:   gpt4oModel,
		Usage:   deepseekResp.Usage,
	}

	openAIResp.Choices = make([]struct {
		Index        int     `json:"index"`
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	}, len(deepseekResp.Choices))

	for i, choice := range deepseekResp.Choices {
		openAIResp.Choices[i] = struct {
			Index        int     `json:"index"`
			Message      Message `json:"message"`
			FinishReason string  `json:"finish_reason"`
		}{
			Index:        choice.Index,
			Message:      choice.Message,
			FinishReason: choice.FinishReason,
		}

		if len(choice.Message.ToolCalls) > 0 {
			debugLog("Processing %d tool calls in choice %d", len(choice.Message.ToolCalls), i)
			for j, tc := range choice.Message.ToolCalls {
				debugLog("Tool call %d: %+v", j, tc)
				if tc.Function.Name == "" {
					debugLog("Warning: Empty function name in tool call %d", j)
					continue
				}
				openAIResp.Choices[i].Message.ToolCalls = append(openAIResp.Choices[i].Message.ToolCalls, tc)
			}
		}
	}

	modifiedBody, err := json.Marshal(openAIResp)
	if err != nil {
		debugLog("Error creating modified response: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	debugLog("Modified response body: %s", string(modifiedBody))

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	w.Write(modifiedBody)
	debugLog("Modified response sent successfully")
}

func copyHeaders(dst, src http.Header) {
	skipHeaders := map[string]bool{
		"Content-Length":    true,
		"Content-Encoding":  true,
		"Transfer-Encoding": true,
		"Connection":        true,
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
	debugLog("Handling models request")
	response := ModelsResponse{
		Object: "list",
		Data: []Model{
			{
				ID:      "gpt-4o",
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "openai",
			},
			{
				ID:      "deepseek-chat",
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "deepseek",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
	debugLog("Models response sent successfully")
}

// ... existing code ...
