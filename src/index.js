import { HfInference } from "@huggingface/inference";
import {config} from 'dotenv'
import { pipeline } from '@xenova/transformers';


config({path: '.env.local'})

function dotProduct(numA, numB) {
  if (numA.length !== numB.length) {
    throw new Error('??')
  }

  let result = 0;

  for (let i = 0; i < numA.length; i += 1){
    result += numA[i] * numB[i];
  }

  return result;
}

async function huggingface_embedding() {
  const hf = new HfInference(process.env.HUGGING_FACE_TOKEN);

  const output = await hf.featureExtraction({
    model: "intfloat/e5-small",
    inputs: "That is a happy person",
  });

  const output_2 = await hf.featureExtraction({
    model: "intfloat/e5-small",
    inputs: "That is a happy person",
  });

  const similarity = dotProduct(output, output_2); 

  console.log(similarity);
}

async function transformer_embedding() {
  let extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  let output = await extractor('This is a simple test.', { pooling: 'mean', normalize: true });
  let output_2 = await extractor('This is a very easy test.', { pooling: 'mean', normalize: true });

  const similarity = dotProduct(output.data, output_2.data); 
  console.log(similarity);
}

transformer_embedding();
 