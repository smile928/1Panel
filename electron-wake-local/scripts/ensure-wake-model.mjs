import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, "..");
const models = path.join(root, "models");
const target = path.join(models, "xiaolanghua.onnx");
const fallback = path.join(models, "hello_deepa.onnx");

if (!fs.existsSync(target)) {
  if (!fs.existsSync(fallback)) {
    console.error("Missing models. Run: npx openwakeword-js-setup");
    process.exit(1);
  }
  fs.copyFileSync(fallback, target);
  console.warn(
    "[wake] Copied placeholder models/hello_deepa.onnx -> models/xiaolanghua.onnx.\n" +
      "Replace xiaolanghua.onnx with a model trained for the phrase「小浪花，小浪花」(see models/README.md)."
  );
} else {
  console.log("[wake] models/xiaolanghua.onnx present.");
}
