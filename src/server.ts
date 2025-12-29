import express from "express";
import { config } from "dotenv";
import { runWorkflow } from "./agent.js";
import type { ChatRequest, ChatResponse } from "./types.js";

// Load environment variables
config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());

// Health check endpoint
app.get("/health", (req, res) => {
    res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// Main chat handler
const chatHandler = async (req: express.Request, res: express.Response) => {
    const requestId = crypto.randomUUID().substring(0, 8);
    console.log(`\n========== NEW REQUEST [${requestId}] ==========`);
    console.log(`[${requestId}] Endpoint: ${req.path}`);
    console.log(`[${requestId}] Timestamp:`, new Date().toISOString());
    console.log(`[${requestId}] Body:`, JSON.stringify(req.body, null, 2));

    try {
        const { message, conversationId, clientId } = req.body as ChatRequest;

        if (!message) {
            console.log(`[${requestId}] ERROR: Missing message field`);
            return res.status(400).json({
                error: "Missing required field: message",
                response: "",
                conversationId: conversationId || crypto.randomUUID()
            } as ChatResponse);
        }

        console.log(`[${requestId}] Processing message: "${message}"`);
        console.log(`[${requestId}] ConversationId: ${conversationId || 'not provided'}`);
        console.log(`[${requestId}] ClientId: ${clientId || 'not provided'}`);

        // Run the agent workflow
        console.log(`[${requestId}] Calling runWorkflow...`);
        const result = await runWorkflow({
            input_as_text: message,
            conversationId: conversationId,
            clientId: clientId
        });
        console.log(`[${requestId}] Workflow completed`);

        const response: ChatResponse = {
            response: result.output_text ? `${result.output_text}` : JSON.stringify(result),
            classification: result.classification,
            conversationId: conversationId || crypto.randomUUID()
        };

        console.log(`[${requestId}] === FINAL RESPONSE ===`);
        console.log(`[${requestId}] Classification: ${response.classification}`);
        console.log(`[${requestId}] Response: ${response.response}`);

        res.json(response);
        console.log(`[${requestId}] Response sent successfully`);
        console.log(`========== END REQUEST [${requestId}] ==========\n`);
    } catch (error) {
        console.error(`[${requestId}] ERROR:`, error);
        return res.status(500).json({
            error: error instanceof Error ? error.message : "Internal server error",
            response: "Lo siento, hubo un error procesando tu mensaje. Por favor intenta de nuevo.",
            conversationId: crypto.randomUUID()
        } as ChatResponse);
    }
};

// Endpoints
app.post("/api/chat", chatHandler);
app.post("/webhook", chatHandler);

// Start server
app.listen(PORT, () => {
    console.log(`🚀 CEA Agent Server running on port ${PORT}`);
    console.log(`📍 Health check: http://localhost:${PORT}/health`);
    console.log(`💬 Chat endpoint: http://localhost:${PORT}/api/chat`);
    console.log(`🔗 Webhook endpoint: http://localhost:${PORT}/webhook`);
});
