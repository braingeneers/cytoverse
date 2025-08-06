// src/types/onnxruntime-web.d.ts

// Attempt to re-export the types from their known location.
// This tells TypeScript: "When I import 'onnxruntime-web',
// get the types from this specific file within the package."
declare module 'onnxruntime-web' {
  // Adjust the path if necessary based on the actual structure
  // within node_modules/onnxruntime-web, but 'types' is common.
  export { Tensor, InferenceSession, env } from 'onnxruntime-web/types.d.ts'
}
