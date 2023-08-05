import { HfInference } from "@huggingface/inference";
import {config} from 'dotenv'

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

async function main() {
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

main();
