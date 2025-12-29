import { runWorkflow } from "./agent";
import dotenv from "dotenv";

// Load environment variables
dotenv.config();

async function main() {
    console.log("Starting test of TICKET CREATION (Fuga)...");

    // Explicit input to trigger immediate ticket creation
    // We use a detailed input to satisfy all agent requirements (location, type, gravity)
    const input = "Reporto una fuga grave de agua potable en la calle 5 de Mayo #123, Colonia Centro. Es en la vía pública, sale mucha agua a presión. Ubicación exacta: 5 de Mayo #123.";

    console.log(`User Input: "${input}"`);

    try {
        const result = await runWorkflow({
            input_as_text: input,
            conversationId: "test-creation-" + Date.now()
        });

        console.log("\nWorkflow Result:");
        console.log("Classification:", result.classification);
        console.log("Output Text:", result.output_text);

        // Check for success indicators
        if (result.output_text?.includes("CEA-FUG") || result.output_text?.includes("folio")) {
            console.log("\n✅ SUCCESS: It seems a ticket folio was generated.");
        } else {
            console.log("\n⚠️ WARNING: No obvious folio found in output. Check logs or output text.");
        }

    } catch (error) {
        console.error("Workflow failed:", error);
    }
}

main().catch(console.error);
