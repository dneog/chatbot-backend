import { HfInference } from '@huggingface/inference';
import { HuggingFaceStream, StreamingTextResponse } from 'ai';
import { experimental_buildOpenAssistantPrompt } from 'ai/prompts';
 
const Hf = new HfInference('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');
 
export const runtime = 'edge';

export async function POST(req) {
  const { messages } = await req.json();
 
  const response = await Hf.textGenerationStream({
    model: 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
    inputs: experimental_buildOpenAssistantPrompt(messages),
    parameters: {
      max_new_tokens: 200,
      typical_p: 0.2,
      repetition_penalty: 1,
      truncate: 1000,
      return_full_text: false,
    },
  });
 
  // Convert the async generator into a friendly text-stream
  const stream = HuggingFaceStream(response);
  return new StreamingTextResponse(stream);
}
