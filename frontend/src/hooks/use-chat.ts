import React from "react";

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export default function useChat() {
    const [messages, setMessages] = React.useState<ChatMessage[]>([]);

    function addMessage({ id, role, content }: ChatMessage) {
        setMessages((prevMessages) => [...prevMessages, { id, role, content }]);
    }

    function clearMessages() {
        setMessages([]);
    }

    function editMessage(id: string, newContent: string) {
        setMessages((prevMessages) =>
            prevMessages.map((msg) => (msg.id === id ? { ...msg, content: newContent } : msg))
        );
    }

    return {
        messages,
        addMessage,
        clearMessages,
        editMessage,
    }
}