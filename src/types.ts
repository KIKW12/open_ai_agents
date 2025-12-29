export interface ChatRequest {
    message: string;
    conversationId?: string;
    clientId?: string;
}

export interface ChatResponse {
    response: string;
    classification?: string;
    conversationId: string;
    error?: string;
}

export type TicketType =
    | "fuga"              // FUG - Fugas/Leaks
    | "aclaraciones"      // ACL - Clarifications
    | "pagos"             // PAG - Payments
    | "lecturas"          // LEC - Meter readings
    | "revision_recibo"   // REV - Receipt review
    | "recibo_digital"    // DIG - Digital receipt
    | "urgente";          // URG - Urgent (human advisor)

export interface WorkflowInput {
    input_as_text: string;
    conversationId?: string;
    clientId?: string;
}

export interface WorkflowOutput {
    output_text?: string;
    classification?: string;
    safe_text?: string;
    pii?: { failed: boolean; detected_counts: string[] };
    moderation?: { failed: boolean; flagged_categories: string[] };
    jailbreak?: { failed: boolean };
    [key: string]: unknown;
}
