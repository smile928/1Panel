import { app, BrowserWindow, ipcMain } from "electron";
import path from "node:path";
import { pathToFileURL } from "node:url";

function modelsRoot(): string {
  return path.join(app.getAppPath(), "models");
}

function toFileUrl(p: string): string {
  return pathToFileURL(p).href;
}

function wasmDirUrl(modelsDir: string): string {
  return pathToFileURL(path.join(modelsDir, "ort.mjs")).href.replace(/ort\.mjs$/i, "");
}

function registerIpc(): void {
  ipcMain.removeHandler("wake:get-model-paths");
  ipcMain.handle("wake:get-model-paths", () => {
    const modelsDir = modelsRoot();
    return {
      modelsDir: toFileUrl(modelsDir),
      melspectrogram: toFileUrl(path.join(modelsDir, "melspectrogram.onnx")),
      embedding: toFileUrl(path.join(modelsDir, "embedding_model.onnx")),
      sileroVad: toFileUrl(path.join(modelsDir, "silero_vad.onnx")),
      wakeword: toFileUrl(path.join(modelsDir, "xiaolanghua.onnx")),
      wasmDir: wasmDirUrl(modelsDir),
    };
  });
}

function broadcast(channel: string, ...args: unknown[]): void {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send(channel, ...args);
  }
}

ipcMain.on("wake:detected", () => {
  /* reserved: hook OS-level actions here */
});

ipcMain.on("wake:barge-in", () => {
  broadcast("wake:stop-tts");
});

function createWindow(): void {
  const win = new BrowserWindow({
    width: 720,
    height: 640,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  win.loadFile(path.join(__dirname, "renderer", "index.html"));
}

app.whenReady().then(() => {
  registerIpc();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
