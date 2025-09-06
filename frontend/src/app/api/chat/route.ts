import { NextRequest } from 'next/server';

// This tells Vercel to use the Edge runtime, which is required for streaming
export const runtime = 'edge';

export async function POST(req: NextRequest) {
    try {
        const { message } = await req.json();

        // IMPORTANT: Replace with your actual Cerebras endpoint and API key
        const CEREBRAS_API_URL = 'https://api.cerebras.ai/v1/chat/completions';
        const CEREBRAS_API_KEY = process.env.CEREBRAS_API_KEY;
        const MODEL_NAME = 'llama3.1-8b'

        if (!CEREBRAS_API_KEY) {
            return new Response('Cerebras API key is not set.', { status: 500 });
        }

        const response = await fetch(CEREBRAS_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${CEREBRAS_API_KEY}`,
            },
            body: JSON.stringify({
                model: MODEL_NAME, // Specify your model
                messages: [{ role: 'user', content: message }],
                stream: true, // Enable streaming
            }),
        });

        // Check for errors from the Cerebras API
        if (!response.ok) {
            const errorText = await response.text();
            return new Response(`Error from Cerebras API: ${errorText}`, { status: response.status });
        }

        // Return the streaming response directly to the client
        return new Response(response.body, {
            headers: {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                Connection: 'keep-alive',
            },
        });
    } catch (error) {
        console.error('Error in chat API route:', error);
        return new Response('Internal Server Error', { status: 500 });
    }
}