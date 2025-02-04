package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/joho/godotenv"
	"golang.org/x/net/http2"
)

const (
	ollamaEndpoint     = "http://localhost:11434/api"
	defaultModel       = "llama2"
	deepseekChatModel  = "michaelneale/deepseek-r1-goose"
	deepseekCoderModel = "deepseek-coder"
	gpt4oModel         = "gpt-4o"
)

// Configuration structure
type Config struct {
	endpoint string
	model    string
}

var activeConfig Config

func init() {
	// Load .env file
	log.Printf("Variant: OLLAMA")
	if err := godotenv.Load(); err != nil {
		log.Printf("Warning: .env file not found or error loading it: %v", err)
	}

	// Get custom Ollama endpoint if specified
	customEndpoint := os.Getenv("OLLAMA_API_ENDPOINT")
	if customEndpoint != "" {
		activeConfig.endpoint = customEndpoint
	} else {
		activeConfig.endpoint = ollamaEndpoint
	}

	// Get custom Ollama endpoint if specified
	modelenv := os.Getenv("DEFAULT_MODEL")
	if modelenv != "" {
		activeConfig.model = modelenv
	} else {
		//no environment set so check for command line argument
		modelFlag := defaultModel // default value
		for i, arg := range os.Args {
			if arg == "-model" && i+1 < len(os.Args) {
				modelFlag = os.Args[i+1]
			}
		}
		activeConfig.model = modelFlag
	}

	log.Printf("Initialized with model: %s using endpoint: %s", activeConfig.model, activeConfig.endpoint)
}

// OpenAI compatible structures
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

// Ollama specific structures
type OllamaRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream"`
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

type OllamaResponse struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Message   struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	Done bool `json:"done"`
}

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds | log.Lshortfile)

	server := &http.Server{
		Addr:    ":9000",
		Handler: http.HandlerFunc(proxyHandler),
	}

	// Enable HTTP/2 support
	http2.ConfigureServer(server, &http2.Server{})

	log.Printf("Starting Ollama proxy server on %s", server.Addr)
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
	log.Printf("Received request: %s %s", r.Method, r.URL.Path)

	if r.Method == "OPTIONS" {
		enableCors(w)
		return
	}

	enableCors(w)

	switch r.URL.Path {
	case "/v1/chat/completions":
		handleChatCompletions(w, r)
	case "/v1/models":
		handleModelsRequest(w)
	default:
		http.Error(w, "Not found", http.StatusNotFound)
	}
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var chatReq ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&chatReq); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// If the model is not specified, use the default model
	if chatReq.Model == "" {
		chatReq.Model = activeConfig.model
	}
	// Convert to Ollama request format
	ollamaReq := OllamaRequest{
		Model:    chatReq.Model,
		Messages: chatReq.Messages,
		Stream:   chatReq.Stream,
	}

	if chatReq.Temperature != nil {
		ollamaReq.Temperature = *chatReq.Temperature
	}
	if chatReq.MaxTokens != nil {
		ollamaReq.MaxTokens = *chatReq.MaxTokens
	}

	// Create Ollama request
	ollamaReqBody, err := json.Marshal(ollamaReq)
	if err != nil {
		log.Printf("ERROR: failed to marshal ollama request: %s", err.Error())

		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Send request to Ollama
	ollamaResp, err := http.Post(
		fmt.Sprintf("%s/chat", activeConfig.endpoint),
		"application/json",
		bytes.NewBuffer(ollamaReqBody),
	)
	if err != nil {
		log.Printf("ERROR: POST failed: %s", err.Error())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer ollamaResp.Body.Close()

	if chatReq.Stream {
		handleStreamingResponse(w, r, ollamaResp)
	} else {
		handleRegularResponse(w, ollamaResp)
	}
}

func handleStreamingResponse(w http.ResponseWriter, r *http.Request, resp *http.Response) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err != io.EOF {
				log.Printf("Error reading stream: %v", err)
			}
			break
		}

		var ollamaResp OllamaResponse
		if err := json.Unmarshal(line, &ollamaResp); err != nil {
			log.Printf("Error unmarshaling response: %v", err)
			continue
		}

		// Convert to OpenAI format
		openAIResp := map[string]interface{}{
			"id":      "chatcmpl-" + time.Now().Format("20060102150405"),
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   activeConfig.model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"role":    "assistant",
						"content": ollamaResp.Message.Content,
					},
					"finish_reason": nil,
				},
			},
		}

		if ollamaResp.Done {
			openAIResp["choices"].([]map[string]interface{})[0]["finish_reason"] = "stop"
		}

		if data, err := json.Marshal(openAIResp); err == nil {
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}

		if ollamaResp.Done {
			break
		}
	}
}

func handleRegularResponse(w http.ResponseWriter, resp *http.Response) {
	var ollamaResp OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Convert to OpenAI format
	openAIResp := map[string]interface{}{
		"id":      "chatcmpl-" + time.Now().Format("20060102150405"),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   activeConfig.model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": ollamaResp.Message.Content,
				},
				"finish_reason": "stop",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(openAIResp)
}

func handleModelsRequest(w http.ResponseWriter) {
	// For simplicity, we return a static list of models
	models := ModelsResponse{
		Object: "list",
		Data: []Model{
			{
				ID:      activeConfig.model,
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "ollama",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models)
}

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
