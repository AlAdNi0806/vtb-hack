import {
    SonioxClient,
    type ErrorStatus,
    type RecorderState,
    type Token,
    type TranslationConfig,
} from '@/lib/ario';
import { useCallback, useEffect, useRef, useState } from 'react';

const END_TOKEN = '<end>';

interface UseTranscribeParameters {
    apiKey: string | (() => Promise<string>);
    translationConfig?: TranslationConfig;
    onStarted?: () => void;
    onFinished?: () => void;
    onEndOfSpeech?: (fullText: string) => void;
    onPartialResult?: (finalTokens: Token[], nonFinalTokens: Token[]) => void;
}

type TranscriptionError = {
    status: ErrorStatus;
    message: string;
    errorCode: number | undefined;
};

// useTranscribe hook wraps Soniox speech-to-text-web SDK.
export default function useTranscribe({ apiKey, translationConfig, onStarted, onFinished, onEndOfSpeech, onPartialResult }: UseTranscribeParameters) {
    const sonioxClient = useRef<SonioxClient | null>(null);

    if (sonioxClient.current == null) {
        sonioxClient.current = new SonioxClient({
            apiKey: apiKey,
        });
    }

    const [state, setState] = useState<RecorderState>('Init');
    const [finalTokens, setFinalTokens] = useState<Token[]>([]);
    const [nonFinalTokens, setNonFinalTokens] = useState<Token[]>([]);
    const [error, setError] = useState<TranscriptionError | null>(null);

    const startTranscription = useCallback(async () => {
        setFinalTokens([]);
        setNonFinalTokens([]);
        setError(null);

        // First message we send contains configuration. Here we set if we set if we
        // are transcribing or translating. For translation we also set if it is
        // one-way or two-way.
        sonioxClient.current?.start({
            model: 'stt-rt-preview',
            enableLanguageIdentification: true,
            enableSpeakerDiarization: true,
            enableEndpointDetection: true,
            translation: translationConfig || undefined,

            onFinished: onFinished,
            onStarted: onStarted,

            onError: (status: ErrorStatus, message: string, errorCode: number | undefined) => {
                setError({ status, message, errorCode });
            },

            onStateChange: ({ newState }) => {
                setState(newState);
            },

            // When we receive some tokens back, sort them based on their status --
            // is it final or non-final token.
            onPartialResult(result) {
                console.log('Partial result', result);
                const newFinalTokens: Token[] = [];
                const newNonFinalTokens: Token[] = [];
                let endTokenReceived = false;

                for (const token of result.tokens) {
                    // Ignore endpoint detection tokens
                    if (token.text === END_TOKEN) {
                        endTokenReceived = true;
                        continue;
                    }

                    if (token.is_final) {
                        newFinalTokens.push(token);
                    } else {
                        newNonFinalTokens.push(token);
                    }
                }

                const currentFinalTokens = [...finalTokens, ...newFinalTokens];

                if (onPartialResult) {
                    onPartialResult(currentFinalTokens, newNonFinalTokens);
                }

                if (endTokenReceived) {
                    const fullText = currentFinalTokens.map((t) => t.text).join('');
                    if (fullText.trim() && onEndOfSpeech) {
                        onEndOfSpeech(fullText);
                    }
                    setFinalTokens([]); // Reset for next utterance
                } else {
                    setFinalTokens((previousTokens) => [
                        ...previousTokens,
                        ...newFinalTokens,
                    ]);
                }

                setNonFinalTokens(newNonFinalTokens);
            },
        });
    }, [onFinished, onStarted, translationConfig, onEndOfSpeech, finalTokens]);

    const stopTranscription = useCallback(() => {
        sonioxClient.current?.stop();
    }, []);

    useEffect(() => {
        return () => {
            sonioxClient.current?.cancel();
        };
    }, []);

    return {
        startTranscription,
        stopTranscription,
        state,
        finalTokens,
        nonFinalTokens,
        error,
    };
}